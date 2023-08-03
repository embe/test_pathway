#!/bin/bash
# Pathway http monitoring set port
if [[ "$RUNNER_NAME" == "ec2-mac2-1" ]]; then
  export PATHWAY_MONITORING_HTTP_PORT=20001 
elif [[ "$RUNNER_NAME" == "ec2-mac2-2" ]]; then 
  export PATHWAY_MONITORING_HTTP_PORT=20002 
elif [[ "$RUNNER_NAME" == "ec2-mac2-3" ]]; then 
  export PATHWAY_MONITORING_HTTP_PORT=20003
elif [[ "$RUNNER_NAME" == "ec2-mac2-4" ]]; then
  export PATHWAY_MONITORING_HTTP_PORT=20004
fi