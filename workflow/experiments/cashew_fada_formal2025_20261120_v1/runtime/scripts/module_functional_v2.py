"""Forward hooks on a fixed ordered panel prefix; no extra forward or RNG use."""
import torch

MODULES = {'embedding':'Swin.patch_embed', **{'encoder_stage%d'%(i+1):'Swin.layers.%d'%i for i in range(4)},
           'pyramid_fusion':'Swin.head', 'fusion':'fuse_conv',
           'refiner0':'a3_spatial_refiner.0', 'refiner1':'a3_spatial_refiner.1', 'decoder':'decoder'}
FIELDS = ('activation_rms','activation_mean','sample_elements','observed_batches')
KEYS = {'functional_'+c+'_'+f for c in MODULES for f in FIELDS}

class FunctionalObserver:
    def __init__(self, model, max_batches=2):
        self.model=model; self.max_batches=max_batches; self.values={}; self.handles=[]
    def __enter__(self):
        modules=dict(self.model.named_modules())
        for c,name in MODULES.items():
            if name not in modules:continue
            self.values[c]=[0.,0.,0,0]
            def hook(module,args,output,key=c):
                row=self.values[key]
                if row[3]>=self.max_batches:return
                while isinstance(output,(tuple,list)):output=output[0]
                if not isinstance(output,torch.Tensor):return
                v=output.detach().float().reshape(-1)
                # Bounded, deterministic sample of activations on fixed inputs.
                stride=max(1,v.numel()//4096);v=v[::stride][:4096]
                row[0]+=float(v.double().square().sum());row[1]+=float(v.double().sum())
                row[2]+=v.numel();row[3]+=1
            self.handles.append(modules[name].register_forward_hook(hook))
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()
    def result(self):
        out={}
        for c,(sq,total,n,batches) in self.values.items():
            vals=dict(activation_rms=(sq/n)**.5 if n else None,activation_mean=total/n if n else None,
                      sample_elements=n,observed_batches=batches)
            out.update({'functional_'+c+'_'+k:v for k,v in vals.items()})
        return out
