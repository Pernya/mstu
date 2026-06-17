#!/usr/bin/env bash
set -e

superset run -h 0.0.0.0 -p 8088 --with-threads
