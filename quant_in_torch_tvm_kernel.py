from PIL import Image
import numpy as np
import torch
from torchvision.models.quantization import resnet18 as qresnet18
from time import time
import tvm
from tvm import relay
import os
import numpy as np
from collections import namedtuple

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
from time import time
from tvm.relay import quantize as qtz

import logging

logging.basicConfig(level=logging.INFO)

Config = namedtuple('Config',
                    ['model', 'nbit_input', 'dtype_input', 'nbit_weight', 'dtype_weight', 'nbit_output', 'dtype_output', 'global_scale', 'batch_size'])

def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration    exit(0)
    model(inp)
    torch.quantization.convert(model, inplace=True)

target = "llvm -mcpu=core-avx2"
batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)
tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "early_stopping": True,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}


def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)



def tune_and_evaluate(tuning_opt, cfg, target, ctx, log_file):
    qconfig = qtz.qconfig(skip_conv_layers=[0],
                          nbit_input=cfg.nbit_input,
                          nbit_weight=cfg.nbit_input,
                          global_scale=cfg.global_scale,
                          dtype_input=cfg.dtype_input,
                          dtype_weight=cfg.dtype_weight,
                          dtype_activation=cfg.dtype_output,
                          debug_enabled_ops=None)
    inp = np.random.rand(1, 3, 224, 224).astype(np.float32)
    qmodel = qresnet18(pretrained=False).eval()
    pt_inp = torch.from_numpy(inp)
    quantize_model(qmodel, pt_inp)
    script_module = torch.jit.trace(qmodel, pt_inp).eval()

    with torch.no_grad():
        pt_result = script_module(pt_inp).numpy()

    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    input_shapes = [(input_name, (1, 3, 224, 224))]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    mod = relay.quantize.quantize(mod, params=params)

    # net = mod['main']
    # start_time = time()
    # with relay.build_config(opt_level=3):
    #     qfunc = relay.quantize.prerequisite_optimize(net, params=params)
    # exit(0)
    tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target="llvm")

    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params
    )

    # for i in range(len(tasks)):
    #     op_name = tasks[i].workload[0]
    #     if op_name == 'conv2d_NCHWc.x86':
    #         func_create = 'topi_x86_conv2d_NCHWc_int8'
    #     elif op_name == 'depthwise_conv2d_nchw':
    #         func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
    #     else:
    #         continue
    #         # print ("Tuning {} is not supported on x86")
    #         # raise ValueError("Tuning {} is not supported on x86".format(op_name))
    #
    #     print ( "[Create Task %2d/%2d (%s, %s) ] " % (i+1, len(tasks), tasks[i].name, tasks[i].workload[0]))
    #
    #     tsk = autotvm.task.create(func_create, args=tasks[i].args,
    #                                 target=tasks[i].target)
    #     tsk.workload = tasks[i].workload
    #     tasks[i] = tsk
    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)
    #tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    with autotvm.apply_history_best(log_file):
        logging.info("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        dev = tvm.cpu()

        # export library
        data_tvm = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype(dtype))
        module = runtime.GraphModule(lib["default"](dev))
        module.set_input(input_name, data_tvm)

        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond

        script_module = torch.jit.trace(qmodel, pt_inp).eval()

        # with torch.no_grad():
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        # load parameters
        # module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
        # data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
        # module.set_input('data', data_tvm)
        # module.set_input(**params)
        #
        # # evaluate
        # logging.info("Evaluate inference time cost...")
        # ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=60)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # logging.info("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

    # compile kernels with graph-level best records
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    #     print("Compile...")
    #     with tvm.transform.PassContext(opt_level=3):
    #         lib = relay.build_module.build(mod, target=target, params=params)
    #
    #     # upload parameters to device
    #     dev = tvm.cpu()
    #     data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    #     module = runtime.GraphModule(lib["default"](dev))
    #     module.set_input(input_name, data_tvm)
    #
    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
    #     prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #     print(
    #         "Mean inference time (std dev): %.2f ms (%.2f ms)"
    #         % (np.mean(prof_res), np.std(prof_res))
    #     )

if __name__ == "__main__":
    target = 'llvm -mcpu=core-avx2'
    ctx = tvm.cpu()

    configs = [
        Config('resnet18', nbit_input=8, dtype_input='uint8',  nbit_weight=8, dtype_weight="int8", nbit_output=32, dtype_output='int32', global_scale=8.0,
               batch_size=1),
        Config('resnet18', nbit_input=16, dtype_input='int16', nbit_weight=8, dtype_weight="int8", nbit_output=16, dtype_output='int16',
               global_scale=8.0, batch_size=1),
        # Config('mobilenetv2_1.0', nbit_input=8, dtype_input='int8', nbit_output=8, dtype_output='int8', global_scale=4.0, batch_size=1),
        # Config('mobilenetv2_1.0', nbit_input=16, dtype_input='int16', nbit_output=16, dtype_output='int16', global_scale=4.0, batch_size=1),
    ]
    for config in configs:
        logging.info('Start testing for %s', config.model)

        log_file = "%s_%s.log" % (config.model, config.dtype_input)
        if os.path.exists(log_file):
            os.remove(log_file)
        tuning_option = {
            'log_filename': log_file,
            'tuner': 'xgb',
            'early_stopping': True,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                # runner=autotvm.RPCRunner(
                #     '1080ti',  # change the device key to your key
                #     '0.0.0.0', 9190,
                #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
            ),
        }

        tune_and_evaluate(tuning_option, config, target, ctx, log_file)
