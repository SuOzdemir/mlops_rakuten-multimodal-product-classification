#!/bin/sh
set -eu

cd /project
exec python -m promotion.promote
