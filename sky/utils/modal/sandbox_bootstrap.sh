#!/bin/bash
# Prepares a new Modal sandbox for SkyPilot. Must be safe to run again: SkyPilot
# runs it on every `sky launch`, including onto an existing cluster.
#
# Modal images are often minimal (debian_slim has no rsync, curl or sudo), and
# the steps after this one need these tools: file mounts need rsync, and
# SkyPilot's runtime installation needs the rest. Kubernetes does the same at
# pod start, in kubernetes-ray.yml.j2.
set -u

sudo_dir="${SKYPILOT_MODAL_SUDO_DIR:-/usr/local/bin}"
if [ "$(id -u)" -eq 0 ] && ! command -v sudo >/dev/null 2>&1; then
    # Root needs no privilege change, and installing the real sudo would need
    # the network. SkyPilot's commands call `sudo` without a tty.
    cat > "$sudo_dir/sudo" <<'SHIM'
#!/bin/sh
# SkyPilot: already root, so run the command directly.
while [ $# -gt 0 ]; do
    case "$1" in
        -u|-g|-p|-C|-D|-h|-R|-T|-U) shift 2 ;;
        --) shift; break ;;
        -*) shift ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || exit 0
exec env "$@"
SHIM
    chmod +x "$sudo_dir/sudo"
fi

# <command it provides>:<apt package>:<dnf/yum package>:<apk package>
tools='rsync:rsync:rsync:rsync
curl:curl:curl:curl
wget:wget:wget:wget
gcc:gcc:gcc:gcc
patch:patch:patch:patch
lspci:pciutils:pciutils:pciutils
nc:netcat-openbsd:nmap-ncat:netcat-openbsd'
required='rsync curl patch'

if command -v apt-get >/dev/null 2>&1; then
    field=2
elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
    field=3
elif command -v apk >/dev/null 2>&1; then
    field=4
else
    field=
fi

missing=
for line in $tools; do
    command -v "${line%%:*}" >/dev/null 2>&1 && continue
    [ -n "$field" ] && missing="$missing $(echo "$line" | cut -d: -f"$field")"
done

if [ -n "$missing" ]; then
    echo "Installing:$missing"
    case "$field" in
        2)
            export DEBIAN_FRONTEND=noninteractive
            sudo apt-get update -q
            # shellcheck disable=SC2086
            sudo apt-get install -y -q --no-install-recommends $missing
            ;;
        3)
            pkg_mgr=$(command -v dnf || command -v yum)
            # shellcheck disable=SC2086
            sudo "$pkg_mgr" install -y $missing
            ;;
        4)
            # shellcheck disable=SC2086
            sudo apk add --no-cache $missing
            ;;
    esac
fi

status=0
for tool in $required; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "SkyPilot needs '$tool' in the Modal sandbox and could not install it." >&2
        status=1
    fi
done
exit $status
