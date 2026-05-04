GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider azure --model o3-mini --url "https://ai-agcohneastus2hub899344470894.openai.azure.com/" --key "REDACTED"  --seed 123 --repeat "0:3"  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
