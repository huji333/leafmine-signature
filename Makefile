IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data
PIPELINE_ARGS ?=

.PHONY: setup build run process_segmented remove_data clean

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
