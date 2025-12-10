IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data

.PHONY: setup build run clean

setup:
	$(MAKE) build

build:
	docker build -t $(IMAGE) .

run: build
	@mkdir -p $(DATA_DIR)
	docker run --rm -it -p 7860:7860 \
		-v $(DATA_DIR):/app/data \
		$(IMAGE)

clean:
	docker image rm $(IMAGE) || true
