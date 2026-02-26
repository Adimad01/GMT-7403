GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider azure --model o4-mini-2025-04-16  --url "$$AZURE_OPENAI_ENDPOINT_4" --key "$$AZURE_OPENAI_API_KEY_4"  --seed 123  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
