import sys
import os

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
print("parent_dir = ", parent_dir)
sys.path.append(parent_dir)


from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import load_start_checkpoint
import torch
import torch.nn as nn
import openvino
from openvino.tools.ovc import convert_model


def patch_nan_to_zero(model):
    from openvino.runtime import opset10 as opset  # opset10 has IsNaN + Select 
    def nan_to_zero(t):
        # Build: select(is_nan(t), 0(t.dtype), t)
        zero = opset.constant(0, dtype=t.get_element_type())
        return opset.select(opset.is_nan(t), zero, t)
    patched = 0
    for node in list(model.get_ops()):
        if node.get_type_name() == "Divide": #and "band_split" in node.get_friendly_name():
            print("Adding NaN-to-Zero after:", node.get_friendly_name())
            out = node.output(0)

            # 1) Freeze the current consumer list (BEFORE inserting the new node)
            original_consumers = list(out.get_target_inputs())

            # 2) Create the nan→zero patch that reads from 'out'
            fixed = nan_to_zero(out)
            fixed.set_friendly_name(node.get_friendly_name() + "__nan2zero")

            # 3) Rewire only the ORIGINAL consumers to 'fixed'
            for inp in original_consumers:
                # (No need to check; 'fixed' wasn't a consumer yet when we captured the list)
                inp.replace_source_output(fixed.output(0))

            patched += 1

    print(f"Patched {patched} Divide nodes.")


def convert_mdx23c(model, config):
    chunk_size = config.audio.chunk_size

    print("chunk_size = ", chunk_size)

    arr = torch.zeros([1, 2, chunk_size], dtype=torch.float32)

    pre_fwd_out = model.pre_forward(arr)
    fwd_out = model.fwd(pre_fwd_out)


    with torch.inference_mode():

        #Converting the 'pre_forward' (stft routine) actually works fine, but C++
        # implementation just used native libtorch instead.
        '''
        class PreWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, arr):
                x = self.model.pre_forward(arr)

                return x
        print("Converting pre-model..")
        premodel = PreWrapper(model)
        with torch.no_grad():
            ov_model = convert_model(premodel, example_input=arr)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"x_in"})
            ov_model.outputs[0].get_tensor().set_names({"x_out"})
            ov_model.reshape(arr.shape)
            ov_model.validate_nodes_and_infer_types()
            openvino.runtime.save_model(ov_model, "mdx23c_pre.xml", compress_to_fp16=True)
        print("done converting pre-model...")
       '''

        class FwdWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.fwd(x)

        print("Converting fwd model (pytorch->openvino)..")
        fwdmodel = FwdWrapper(model)
        with torch.no_grad():
            ov_model = convert_model(fwdmodel, example_input=pre_fwd_out)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"x_in"})
            ov_model.outputs[0].get_tensor().set_names({"x_out"})
            ov_model.reshape(pre_fwd_out.shape)
            ov_model.validate_nodes_and_infer_types()
            openvino.runtime.save_model(ov_model, "mdx23c_fwd.xml", compress_to_fp16=True)
        print("done converting fwd model.")

        # seems to be some issue in post_forward for conversion.
        # conversion itself seems to work fine, but failures occur during 'reshape', after conversion
        # TODO: Raise this issue to OpenVINO team.
        '''
        class PostWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                x = self.model.post_fwd(x)
                return x

        print("converting post-model...")
        postmodel = PostWrapper(model)
        with torch.no_grad():
            ov_model = convert_model(postmodel, example_input=fwd_out)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"x_in"})
            ov_model.outputs[0].get_tensor().set_names({"x_out"})
            ov_model.reshape(fwd_out.shape)
            ov_model.validate_nodes_and_infer_types()
            openvino.runtime.save_model(ov_model, "mdx23c_post.xml", compress_to_fp16=True)
        print("done converting post-model...")
        '''

def convert_apollo(model, config):
    print("convert_apollo start..")
    chunk_size = config.audio.chunk_size

    arr = torch.zeros([1, 2, chunk_size], dtype=torch.float32)

    print("arr.shape = ", arr.shape)
    B, nch, nsample = arr.shape

    spec = model.pre_forward(arr)

    with torch.inference_mode():
        class FwdWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, spec):
                return self.model.fwd(spec, B, nch, nsample)

        print("Converting fwd model (pytorch->openvino)..")
        fwdmodel = FwdWrapper(model)
        with torch.no_grad():
            ov_model = convert_model(fwdmodel, example_input=spec)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"spec"})
            ov_model.outputs[0].get_tensor().set_names({"est_spec"})
            ov_model.reshape(spec.shape)
            ov_model.validate_nodes_and_infer_types()
            openvino.runtime.save_model(ov_model, "apollo_fwd.xml", compress_to_fp16=True)
        print("done converting fwd model.")

def convert_mel_band_roformer(model, config):
    print("convert_mel_band_roformer start..")
    chunk_size = config.audio.chunk_size
    print("chunk_size = ", chunk_size)
    
    arr = torch.zeros([1, 2, chunk_size], dtype=torch.float32)
    batch, channels, raw_audio_length = arr.shape
    
    # run pre & fwd to generate some dummy tensors to use for later conversion(s)
    stft_repr, stft_window, istft_length = model.pre_forward(arr)
    masks = model.fwd(stft_repr)
    
    with torch.inference_mode():
        class PreWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, arr):
                stft_repr, stft_window, istft_length = self.model.pre_forward(arr)
                
                return stft_repr, stft_window
        print("Converting pre-model..")       
        premodel = PreWrapper(model)
        with torch.no_grad():
            ov_model = convert_model(premodel, example_input=arr)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"arr"})
            ov_model.outputs[0].get_tensor().set_names({"stft_repr"})
            ov_model.outputs[1].get_tensor().set_names({"stft_window"})
            openvino.runtime.save_model(ov_model, "mel_band_pre.xml", compress_to_fp16=True)
        print("done converting pre-model...")  
        
        class FwdWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, stft_repr):
                return self.model.fwd(stft_repr)
                
        print("Converting fwd model (pytorch->onnx->openvino)..")
        fwdmodel = FwdWrapper(model)
        with torch.no_grad():
            torch.onnx.export(fwdmodel, (stft_repr), "fwdmodel.onnx", input_names=["stft_repr"], output_names=["masks"])
            ov_model = convert_model("fwdmodel.onnx")
            #ov_model = convert_model(fwdmodel, example_input=stft_repr)
            # We insert some nan-to-zero operations after Divide ops.
            patch_nan_to_zero(ov_model)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"stft_repr"})
            ov_model.outputs[0].get_tensor().set_names({"masks"})
            openvino.runtime.save_model(ov_model, "mel_band_fwd.xml", compress_to_fp16=True)
        print("done converting fwd model.")
        
        class PostWrapper(nn.Module):
            def __init__(self, model, batch, channels):
                super().__init__()
                self.model = model
                self.batch = batch
                self.channels = channels
            
            def forward(self, stft_repr, masks, stft_window):
                recon = self.model.post_forward(stft_repr, masks, stft_window, None, self.batch, self.channels)
                return recon
        
        print("converting post-model...")
        postmodel = PostWrapper(model, batch, channels)
        with torch.no_grad():
            dummy_input = {"stft_repr": stft_repr, "masks": masks, "stft_window": stft_window}
            ov_model = convert_model(postmodel, example_input=dummy_input)
            ov_model.validate_nodes_and_infer_types()
            ov_model.inputs[0].get_tensor().set_names({"stft_repr"})
            ov_model.inputs[1].get_tensor().set_names({"masks"})
            ov_model.inputs[2].get_tensor().set_names({"stft_window"})
            ov_model.outputs[0].get_tensor().set_names({"recon"})
            ov_model.validate_nodes_and_infer_types()
            openvino.runtime.save_model(ov_model, "mel_band_post.xml", compress_to_fp16=True)
        print("done converting post-model...")
        
        
def convert_htdemucs(model, config):
    print("convert_htdemucs start..")
    chunk_size = config.training.samplerate * config.training.segment
    
    # we override the chunk_size here for htdemucs, as it seems that this size gives
    # much better peformance when using OpenVINO devices such as GPU and NPU.
    chunk_size = 343980
    
    arr = torch.zeros([1, 2, chunk_size], dtype=torch.float32)
    
    # run pre-fwd to produce dummy tensor input for model conversion.
    x, xt, std, mean, meant, stdt, z, length, length_pre_pad = model.pre_forward(arr)
  
    class FwdWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x, xt):
            return self.model.fwd(x, xt)
    
    fwdmodel = FwdWrapper(model)
    
    print("Converting fwd model (pytorch->onnx->openvino)..")
    with torch.inference_mode():
        with torch.no_grad():
           torch.onnx.export(fwdmodel, (x, xt), "htdemucs_fwd.onnx", input_names=["x", "xt"], output_names=["x_out", "xt_out"], dynamo=True, report=True)
           ov_model = convert_model("htdemucs_fwd.onnx")          
           openvino.runtime.save_model(ov_model, "htdemucs_fwd.xml", compress_to_fp16=True)   
    print("done converting fwd model.")           

    # x, xt = model(x, xt)
    # x = model.post_forward(x, xt, std, mean, meant, stdt, z, length, length_pre_pad)    
    
    
    print("convert_htdemucs end..")

def run():
    args = parse_args_inference(None)
    
    model, config = get_model_from_config(args.model_type, args.config_path)
    
    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')
    
    model = model.to("cpu")

    if type(model).__module__ == 'models.bs_roformer.mel_band_roformer' and type(model).__name__ == 'MelBandRoformer':
        convert_mel_band_roformer(model, config)
    elif type(model).__module__ == 'models.demucs4ht' and type(model).__name__ == 'HTDemucs':
        convert_htdemucs(model, config)
    elif type(model).__module__ == 'models.look2hear.models.apollo' and type(model).__name__ == 'Apollo':
        convert_apollo(model, config)
    elif type(model).__module__ == 'models.mdx23c_tfc_tdf_v3' and type(model).__name__ == 'TFC_TDF_net':
        convert_mdx23c(model, config)
    else:
        print("This conversion script does not yet have support for model of type = ", type(model))
    
if __name__ == "__main__":
    run()