# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

class _scheduler(object):
    def __init__(self, last_epoch=-1, verbose=False):
        self.cnt = last_epoch
        self.verbose = verbose
        self.variable = None
        self.step()

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value

    def get_value(self):
        raise NotImplementedError

    def get_variable(self):
        return self.variable


class StepLR(_scheduler):
    def __init__(self, initValue, period, decay=0.1, endValue=None, last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        super(StepLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue


class StepLRMargin(_scheduler):
    def __init__(self, initValue, period, goalValue, decay=0.1, endValue=None, last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.goalValue = goalValue
        super(StepLRMargin, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.goalValue - (self.goalValue-self.initValue) * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue >= self.endValue:
            return self.endValue
        return tmpValue


class StepResetLR(_scheduler):
    def __init__(self, initValue, period, resetPeriod, decay=0.1, endValue=None,
        last_epoch=-1, verbose=False):
        self.initValue = initValue
        self.period = period
        self.decay = decay
        self.endValue = endValue
        self.resetPeriod = resetPeriod
        super(StepResetLR, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == -1:
            return self.initValue

        numDecay = int(self.cnt/self.period)
        tmpValue =  self.initValue * (self.decay ** numDecay)
        if self.endValue is not None and tmpValue <= self.endValue:
            return self.endValue
        return tmpValue

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value
        if (self.cnt+1) % self.resetPeriod == 0:
            self.cnt = -1