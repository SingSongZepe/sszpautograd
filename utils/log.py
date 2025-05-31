
SINGSONGLOG_F = '[SingSongLog {}]: '
DEBUG_MODE = True

def lany(type: str, *msg):
    if DEBUG_MODE:
        print(SINGSONGLOG_F.format(type.upper()) + ''.join([str(m) for m in msg]))

def ln(*msg):
    if DEBUG_MODE:
        print(SINGSONGLOG_F.format('NOR') + ''.join([str(m) for m in msg]))

def lw(*msg):
    if DEBUG_MODE:
        print(SINGSONGLOG_F.format('WAR') + ''.join([str(m) for m in msg]))
          
def le(*msg):
    if DEBUG_MODE:
        print(SINGSONGLOG_F.format('ERR') + ''.join([str(m) for m in msg]))
