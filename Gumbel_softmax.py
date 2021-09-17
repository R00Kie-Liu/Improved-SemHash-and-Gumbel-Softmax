import torch
from torch import nn 
import torch.nn.functional as F

class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False, temperature=1):
        super().__init__()
        self.hard = hard
        self.temperature = temperature

    def sample_gumbel_noise(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_noise = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_noise

    def gumbel_softmax_sample(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        #dim = logits.size(-1)
        gumble_samples_noise = self.sample_gumbel_noise(logits.data)
        gumble_trick_log_prob_samples = logits + gumble_samples_noise
        soft_samples = F.softmax(gumble_trick_log_prob_samples / self.temperature, 1)
        return soft_samples
    
    def gumbel_softmax(self, logits):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        # logits + gumbel_noise
        if self.training:
            y = self.gumbel_softmax_sample(logits)
        else:
            y = F.softmax(logits,1)

        # Return softmax() or hard()
        if self.hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = (y_hard - y.data) + y
        return y
        
    def forward(self, logits):

        if self.training:
            return self.gumbel_softmax(logits)
        else:
            return self.gumbel_softmax(logits)


if __name__ == "__main__":
    gumbel_trick = GumbleSoftmax(hard=True)
    logits = torch.rand(2, 10)
    print(logits)
    outputs = gumbel_trick(logits)
    print(outputs)
