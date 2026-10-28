"""Extract the unchanged MI-GAN network from its official ONNX pipeline.

The boundary tensor names were inspected in the pinned model. A numerical check
compares this network against the intermediate output of the original graph.
No downloaded Python code or pickle checkpoints are executed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/repair_backend_compare/20261027_migan_fixed'
INPUT='/Concat_23_output_0'
OUTPUT='/model/synthesis/b512/Add_1_output_0'


def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    opts.log_severity_level=3
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])


def main():
    original=OUT/'models/migan_pipeline_v2.onnx'
    expected='6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==expected
    model=onnx.load(original)
    for name,shape in [(INPUT,[1,4,512,512]),(OUTPUT,[1,3,512,512])]:
        model.graph.value_info.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
        model.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shape))
    debug=OUT/'models/official_pipeline_debug.onnx';onnx.save(model,debug)
    core=OUT/'models/migan_core_512.onnx'
    onnx.utils.extract_model(str(debug),str(core),[INPUT],[OUTPUT],check_model=True)
    raw=onnx.load(core)
    original_weights={v.name:v.SerializeToString() for v in model.graph.initializer}
    assert all(original_weights[v.name]==v.SerializeToString() for v in raw.graph.initializer)
    rng=np.random.default_rng(20261027)
    image=rng.integers(25,231,(1,3,512,512),dtype=np.uint8)
    known=np.full((1,1,512,512),255,np.uint8);known[:,:,150:330,170:350]=0
    features,expected_output=session(debug).run([INPUT,OUTPUT],{'image':image,'mask':known})
    keep=known.astype(np.float32)/255
    our_features=np.concatenate([keep-.5,(image.astype(np.float32)*2/255-1)*keep],axis=1)
    actual=session(core).run(None,{INPUT:our_features})[0]
    delta_features=float(np.max(np.abs(features-our_features)))
    delta_output=float(np.max(np.abs(actual-expected_output)))
    assert delta_features<1e-6,delta_features
    assert delta_output<1e-4,delta_output
    record={'source_sha256':expected,'core_sha256':hashlib.sha256(core.read_bytes()).hexdigest(),
            'raw_input_name':INPUT,'raw_output_name':OUTPUT,'weights_byte_identical':True,
            'input_preprocessing_max_abs_difference':delta_features,
            'network_output_max_abs_difference':delta_output,
            'note':'Same unchanged weights/network, extracted only to bypass the official global crop/resize and soft compositing for equal-resolution tiling.'}
    (OUT/'extraction_check.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
