"""Binary Cashew supervision; only newly drawn, valid pixels contribute."""
import torch

VERSION = 'user_polygon_binary_separate_sign_means_v2'

def loss(logits, labels, images=None, *, positives=False, positive_scale=1.):
    """Return negative_mean + positive_scale * positive_mean.

    If the caller applies an outer w_neg>0, set positive_scale=w_pos/w_neg.
    Leave positive_scale=1 to retain the original equal-sign behavior.
    """
    valid = torch.ones_like(labels, dtype=torch.bool)
    if images is not None:
        valid = torch.isfinite(images).all(dim=1) & (images != 0).any(dim=1)
    neg = (labels == 0) & valid
    pos = (labels == 2) & valid if positives else torch.zeros_like(neg)
    # Stable binary probabilities from the unchanged four-class logits.
    normalizer = torch.logsumexp(logits, dim=1)
    negative_nll = normalizer - torch.logsumexp(logits[:, [0,1,3]], dim=1)
    positive_nll = normalizer - logits[:,2]
    zero = logits.sum()*0
    n = negative_nll[neg].mean() if neg.any() else zero
    p = positive_nll[pos].mean() if pos.any() else zero
    result = n+float(positive_scale)*p
    stats = dict(selected_pixels=int(neg.sum()), eligible_pixels=int((neg|pos).sum()),
        pixels=labels.numel(), p2_mean=float(torch.softmax(logits.detach(),dim=1)[:,2][neg].mean()) if neg.any() else 0.,
        negative_pixels=int(neg.sum()), positive_pixels=int(pos.sum()),
        negative_loss=float(n.detach()), positive_loss=float(p.detach()))
    return result,stats

def verify():
    x=torch.randn(2,4,3,3,requires_grad=True)
    y=torch.full((2,3,3),10,dtype=torch.long); y[0,0,0]=0;y[1,1,1]=2
    z,s=loss(x,y,positives=True);z.backward()
    valid=(y==0)|(y==2)
    assert torch.count_nonzero(x.grad.permute(0,2,3,1)[~valid])==0
    assert x.grad[0,2,0,0]>0 and x.grad[1,2,1,1]<0
    a=x.detach().clone();a.permute(0,2,3,1)[~valid]=1000
    assert torch.allclose(loss(a,y,positives=True)[0],z.detach())
    n=loss(x.detach(),y,positives=False)[0]
    assert torch.allclose(n, -torch.log1p(-torch.softmax(x.detach(),dim=1)[0,2,0,0]))
    assert torch.allclose(z.detach()-n, -torch.log(torch.softmax(x.detach(),dim=1)[1,2,1,1]))
    positive_term=z.detach()-n
    for scale in [.25,1.,4.]:
        scaled=x.detach().clone().requires_grad_(True)
        value,_=loss(scaled,y,positives=True,positive_scale=scale)
        assert torch.allclose(value,n+scale*positive_term)
        value.backward()
        assert torch.allclose(scaled.grad[0],x.grad[0])
        assert torch.allclose(scaled.grad[1],scale*x.grad[1])
    empty=x.detach().clone().requires_grad_(True)
    e,_=loss(empty,torch.full_like(y,10),positives=True);e.backward();assert e==0 and torch.count_nonzero(empty.grad)==0
    image=torch.ones(2,64,3,3);image[0,:,0,0]=0
    assert loss(x.detach(),y,image,positives=False)[0]==0
    return dict(state='passed',checks=['ignore pixels have zero logit gradient','ignored logits do not alter loss',
        'negative means non-Cashew','positive increases p2','separate sign means preserve negative coefficient',
        'empty annotation zero loss and gradient','invalid imagery support excluded',
        'independent positive scaling leaves negative gradient unchanged'])

if __name__=='__main__':print(verify())
