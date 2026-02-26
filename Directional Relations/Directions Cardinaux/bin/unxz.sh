#!/usr/bin/env bash
find ../data -type f -name "questions.jsonl.xz" -exec xz -dk {} \;
find ../data -type f -name "answers.jsonl.xz" -exec xz -dk {} \;
