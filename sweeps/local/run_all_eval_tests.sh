#!/bin/bash
# Master script: run context test, then spectral test (sequential on GPU 1)
echo "$(date) Starting eval context test..."
bash ${PROJECT_ROOT:-.}/local_tests/eval_context_test.sh
echo ""
echo "$(date) Starting spectral surgery test..."
bash ${PROJECT_ROOT:-.}/local_tests/eval_spectral_test.sh
echo ""
echo "$(date) ALL TESTS COMPLETE"
