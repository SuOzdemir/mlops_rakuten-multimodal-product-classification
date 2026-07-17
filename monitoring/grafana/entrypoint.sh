#!/bin/sh
set -eu

runtime_dir="/tmp/grafana-provisioning"
rm -rf "${runtime_dir}"
mkdir -p "${runtime_dir}"
cp -R /etc/grafana/provisioning-base/. "${runtime_dir}/"

if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    cp /etc/grafana/provisioning-slack/contactpoints.yaml \
        "${runtime_dir}/alerting/contactpoints.yaml"
    cp /etc/grafana/provisioning-slack/policies.yaml \
        "${runtime_dir}/alerting/policies.yaml"
    echo "Grafana provisioning: Slack alerts enabled."
else
    echo "Grafana provisioning: Slack URL is empty; Slack alerts disabled."
fi

export GF_PATHS_PROVISIONING="${runtime_dir}"
exec /run.sh
