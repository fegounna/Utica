MASTER=faisan.polytechnique.fr
PORT=29500
RDZV=timeserdino_001
WORKDIR=/users/eleves-a/2022/yessin.moakher/timeserdino
ENV=mantisv2

for h in faisan coucou gelinotte jabiru epervier harpie kamiche linotte; do
  case "$h" in
    faisan) r=0;;
    coucou) r=1;;
    gelinotte) r=2;;
    jabiru) r=3;;
    epervier) r=4;;
    harpie) r=5;;
    kamiche) r=6;;
    linotte) r=7;;
  esac

  echo "→ launching $h rank=$r"

  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$h" "bash -lc '
    set -euo pipefail
    tmux kill-session -t dino 2>/dev/null || true

    tmux new -d -s dino \"bash -lc \\\"\
      set -euo pipefail; \
      source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh; \
      conda activate $ENV; \
      cd $WORKDIR; \
      IFACE=\\\$(ip route get 1.1.1.1 2>/dev/null | awk \\\"{for(i=1;i<=NF;i++) if(\\\\\\\$i==\\\\\\\"dev\\\\\\\"){print \\\\\\\$(i+1); exit}}\\\"); \
      IFACE=\\\${IFACE:-eth0}; \
      export NCCL_SOCKET_IFNAME=\\\$IFACE; \
      export GLOO_SOCKET_IFNAME=\\\$IFACE; \
      export MASTER_ADDR=$MASTER; \
      export MASTER_PORT=$PORT; \
      echo [HOST $h] rank=$r iface=\\\$IFACE master=$MASTER:$PORT; \
      torchrun --nnodes=8 --nproc_per_node=1 --node_rank=$r \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT --rdzv_id=$RDZV \
        -m utica.train.train \
      2>&1 | tee -a $WORKDIR/torchrun_rank${r}_\$(hostname -s).log \
    \\\"\"

    tmux ls | grep -q \"^dino:\"
  '"
done
