# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )


class _scheduler(object):

    def __init__(self, last_epoch=-1, verbose=False):
        self.cnt_total = last_epoch
        self.cnt = last_epoch
        self.verbose = verbose
        self.variable = None
        self.step()

    def step(self):
        self.cnt_total += 1
        self.cnt += 1
        value = self.get_value()
        self.variable = value

    def get_value(self):
        raise NotImplementedError

    def get_variable(self):
        return self.variable


class StepLR(_scheduler):

    def __init__(
        self, init_value, period, decay=0.1, end_value=None, last_epoch=-1,
        threshold=0, verbose=False
    ):
        self.init_value = init_value
        self.period = period
        self.decay = decay
        self.end_value = end_value
        self.threshold = threshold
        super(StepLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        cnt = self.cnt - self.threshold
        if cnt < 0:
            return self.init_value

        numDecay = int(cnt / self.period)
        tmpValue = self.init_value * (self.decay**numDecay)
        if self.end_value is not None and tmpValue <= self.end_value:
            return self.end_value
        return tmpValue


class StepLRMargin(_scheduler):

    def __init__(
        self, init_value, period, goal_value, decay=0.1, end_value=None,
        last_epoch=-1, threshold=0, verbose=False
    ):
        self.init_value = init_value
        self.period = period
        self.decay = decay
        self.end_value = end_value
        self.goal_value = goal_value
        self.threshold = threshold
        super(StepLRMargin, self).__init__(last_epoch, verbose)

    def get_value(self):
        cnt = self.cnt - self.threshold
        if cnt < 0:
            return self.init_value

        numDecay = int(cnt / self.period)
        tmpValue = self.goal_value - (self.goal_value - self.init_value
                                     ) * (self.decay**numDecay)
        if self.end_value is not None and tmpValue >= self.end_value:
            return self.end_value
        return tmpValue


class StepLRFixed(_scheduler):

    def __init__(
        self, init_value, period, end_value, step_size=0.1, last_epoch=-1,
        min_step=0, verbose=False
    ):
        self.init_value = init_value
        self.period = period
        self.step_size = step_size
        self.end_value = end_value
        self.min_step = min_step
        super(StepLRFixed, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == 0 or self.cnt_total < self.min_step:
            return self.init_value
        elif self.cnt > self.period:
            self.cnt = 0
            if self.step_size > 0:
                self.variable = min(
                    self.end_value, self.variable + self.step_size
                )
            else:
                self.variable = max(
                    self.end_value, self.variable + self.step_size
                )
        return self.variable


class StepResetLR(_scheduler):

    def __init__(
        self, init_value, period, reset_period, decay=0.1, end_value=None,
        last_epoch=-1, verbose=False
    ):
        self.init_value = init_value
        self.period = period
        self.decay = decay
        self.end_value = end_value
        self.reset_period = reset_period
        super(StepResetLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.init_value

        numDecay = int(self.cnt / self.period)
        tmpValue = self.init_value * (self.decay**numDecay)
        if self.end_value is not None and tmpValue <= self.end_value:
            return self.end_value
        return tmpValue

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value
        if (self.cnt + 1) % self.reset_period == 0:
            self.cnt = -1
