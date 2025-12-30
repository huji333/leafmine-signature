IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data
PIPELINE_ARGS ?=
POLYLINE_ARGS ?=

HOST_ARCH := $(shell /usr/bin/uname -m 2>/dev/null)
DEFAULT_PLATFORM :=
DEFAULT_BUILD_CMD := docker build

ifneq ($(filter arm64 aarch64,$(HOST_ARCH)),)
  DEFAULT_PLATFORM := linux/amd64
  DEFAULT_BUILD_CMD := docker buildx build --load
endif

BUILD_PLATFORM ?= $(DEFAULT_PLATFORM)
RUN_PLATFORM ?= $(DEFAULT_PLATFORM)
BUILD_CMD ?= $(DEFAULT_BUILD_CMD)

ifneq ($(strip $(BUILD_PLATFORM)),)
  BUILD_PLATFORM_FLAG := --platform $(BUILD_PLATFORM)
endif
ifneq ($(strip $(RUN_PLATFORM)),)
  RUN_PLATFORM_FLAG := --platform $(RUN_PLATFORM)
endif

.PHONY: setup build run process_segmented analyze_polylines lint remove_data clean

setup:
	$(MAKE) build

build:
	$(BUILD_CMD) $(BUILD_PLATFORM_FLAG) -t $(IMAGE) .

run: build
	@mkdir -p $(DATA_DIR)
	docker run --rm -it $(RUN_PLATFORM_FLAG) -p 7860:7860 \
		-v $(DATA_DIR):/app/data \
		$(IMAGE)

process_segmented:
ifneq ($(BUILD),)
	$(MAKE) build
endif
	@mkdir -p $(DATA_DIR)
	docker run --rm -it $(RUN_PLATFORM_FLAG) \
		-v $(DATA_DIR):/app/data \
		$(IMAGE) \
		uv run --frozen --no-sync python -m controllers.pipeline --data-dir /app/data $(PIPELINE_ARGS)

analyze_polylines:
ifneq ($(BUILD),)
	$(MAKE) build
endif
	@mkdir -p $(DATA_DIR)
	docker run --rm -it $(RUN_PLATFORM_FLAG) \
		-v $(DATA_DIR):/app/data \
		$(IMAGE) \
		uv run --frozen --no-sync python -m controllers.polyline_signatures --data-dir /app/data $(POLYLINE_ARGS)

lint:
ifneq ($(BUILD),)
	$(MAKE) build
endif
	docker run --rm $(RUN_PLATFORM_FLAG) \
		-v $(CURDIR):/app \
		-w /app \
		$(IMAGE) \
		uv run --frozen --no-sync ruff check .

remove_data:
	@mkdir -p $(DATA_DIR)
	@echo "Pruning generated artifacts under $(DATA_DIR) (set PURGE=1 to remove segmented/ and signatures/ too)."
	@set -e; \
	for entry in $$(/bin/ls -1A $(DATA_DIR) 2>/dev/null); do \
		case $$entry in \
			segmented|signatures) \
				if [ "$$PURGE" = "1" ]; then \
					echo "Removing $(DATA_DIR)/$$entry"; \
					/bin/rm -rf "$(DATA_DIR)/$$entry"; \
				else \
					echo "Preserving $(DATA_DIR)/$$entry"; \
					continue; \
				fi ;; \
			*) \
				echo "Removing $(DATA_DIR)/$$entry"; \
				/bin/rm -rf "$(DATA_DIR)/$$entry"; \
			;; \
		esac; \
	done; \
	if [ "$$PURGE" != "1" ]; then \
		/bin/mkdir -p "$(DATA_DIR)/skeletonized" "$(DATA_DIR)/tmp"; \
	fi

clean:
	docker image rm $(IMAGE) || true
