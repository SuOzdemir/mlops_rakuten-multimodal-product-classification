FROM python:3.12-slim

WORKDIR /project

# Without this, Python block-buffers stdout when it isn't a TTY (i.e.
# always, in a container) -- `dvc repro`'s training subprocess's print()
# output (progress, epoch results) sits in an internal buffer and never
# reaches `docker logs` until it fills (~8KB) or the process exits. A run
# that's actually progressing normally looks identical to a hung one.
ENV PYTHONUNBUFFERED=1

# CPU-only torch — training runs on CPU inside this container
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY trainer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
