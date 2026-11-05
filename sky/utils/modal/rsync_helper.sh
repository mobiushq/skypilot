#!/bin/bash
# The `rsync -e` transport for Modal sandboxes: tunnels rsync's protocol over
# Sandbox.exec, as sky/utils/kubernetes/rsync_helper.sh does over `kubectl exec`.
#
# rsync calls this as `<helper> [-l user] <sandbox_id> rsync --server ...`.
# openrsync passes `user@host` instead, which ModalCommandRunner never produces
# because its destination has no user.
if [ "$1" = "-l" ]; then
    shift 2
fi
sandbox_id=$1
shift

# rsync escapes the remote arguments for a remote shell, because that is what
# ssh gives it, so a shell has to undo that: paths with spaces arrive as
# `my\ dir`. `--norc --noprofile` keeps startup files from writing to stdout,
# which carries the rsync protocol.
remote_script='command -v rsync >/dev/null 2>&1 || { echo "rsync is not installed in the Modal sandbox." >&2; exit 127; }; eval "exec $*"'

exec "${SKYPILOT_MODAL_PYTHON:-python3}" "$(dirname "$0")/exec_helper.py" \
    --sandbox-id "$sandbox_id" \
    --connect-timeout 60 \
    -- /bin/bash --norc --noprofile -c "$remote_script" -- "$@"
