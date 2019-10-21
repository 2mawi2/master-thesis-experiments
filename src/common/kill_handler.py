import signal


class KillHandler:
    """necessary for clean shutdown"""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.catch_kill)
        signal.signal(signal.SIGTERM, self.catch_kill)

    def catch_kill(self, signum, frame):
        self.kill_now = True

    def is_manual_kill(self):
        if self.kill_now:
            if input('Abort training (y/[n])? ') == 'y':
                return True
            else:
                self.kill_now = False
        return False
