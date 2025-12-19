IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data
PIPELINE_ARGS ?=

.PHONY: setup build run process_segmented clean

setup:
	$(MAKE) build

build:
	docker build -t $(IMAGE) .

run: build
	@mkdir -p $(DATA_DIR)
	docker run --rm -it -p 7860:7860 \
		-v $(DATA_DIR):/app/data \
		$(IMAGE)

process_segmented:
ifneq ($(BUILD),)
	$(MAKE) build
endif
	@mkdir -p $(DATA_DIR)
	docker run --rm -it \
		-v $(DATA_DIR):/app/data \
		$(IMAGE) \
		uv run --frozen --no-sync python -m controllers.pipeline --data-dir /app/data $(PIPELINE_ARGS)

clean:
	docker image rm $(IMAGE) || true
