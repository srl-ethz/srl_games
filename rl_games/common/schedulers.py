

class RLScheduler:
    def __init__(self):
        pass

    def update(self,current_lr, entropy_coef, epoch, frames, **kwargs):
        pass

class IdentityScheduler(RLScheduler):
    def __init__(self):
        super().__init__()

     
    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        return current_lr, entropy_coef  


class AdaptiveScheduler(RLScheduler):
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr, entropy_coef         
