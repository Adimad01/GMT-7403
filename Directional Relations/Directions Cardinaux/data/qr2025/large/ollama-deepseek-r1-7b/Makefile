GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider ollama --model "deepseek-r1:7b" --max_tokens 10000 --temperature 0.0 --seed 123 --repeat "0:3" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
