GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --provider deepseek --system-prompt ../system-prompt.txt --url "https://DeepSeek-R1-cdjdw.eastus2.models.ai.azure.com/v1/chat/completions" --key "REDACTED" --repeat "0:3"  --temperature 0.0 --seed 123 -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
