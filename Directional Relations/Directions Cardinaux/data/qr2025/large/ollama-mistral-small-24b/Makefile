GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider ollama --model "mistral-small:24b" --delay 0.1 --temperature 0.0 --seed 123 --repeat "0:3" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
