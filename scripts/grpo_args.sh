# GRPO overrides shared by scripts/jobs/grpo.sge and the CPU dry run (scripts/validate_skyrl_cfg.py).
# Inputs (env): INIT RUN OUT SEED STEPS BATCH NSAMP DBP VLLM_MEM THINK(false|default) TRAIN_DATA VAL_DATA NGPU MAX_GEN MAX_INPUT
# Protocol v1 (PREREG pilot log): verify prompt (data/v1), thinking off, lenient text2sql env (agenticrl.main_grpo)
grpo_args() {
  GRPO_ARGS=(
  "data.train_data=['${TRAIN_DATA:-$ROOT/data/v1/rl_train_nonempty.parquet}']"
  "data.val_data=['${VAL_DATA:-$ROOT/data/v1/spider_dev.parquet}']"
  data.dataloader.num_workers=0
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false
  trainer.algorithm.off_policy_correction.tis_ratio_type=token
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0
  trainer.policy.optimizer_config.lr=1e-6 trainer.policy.optimizer_config.max_grad_norm=0.5
  "trainer.policy.model.path=$INIT"
  trainer.strategy=fsdp trainer.placement.colocate_all=true
  trainer.placement.policy_num_gpus_per_node=${NGPU:-4} trainer.placement.ref_num_gpus_per_node=${NGPU:-4}
  trainer.policy.fsdp_config.cpu_offload=false trainer.ref.fsdp_config.cpu_offload=true
  trainer.flash_attn=true trainer.remove_microbatch_padding=true trainer.gradient_checkpointing=true
  trainer.seed=$SEED trainer.train_batch_size=$BATCH trainer.policy_mini_batch_size=$BATCH
  trainer.micro_train_batch_size_per_gpu=1 trainer.micro_forward_batch_size_per_gpu=1
  trainer.max_prompt_length=6000 trainer.epochs=100 trainer.max_training_steps=$STEPS
  trainer.update_epochs_per_batch=1 trainer.logger=console trainer.enable_ray_gpu_monitor=false
  trainer.resume_mode=none "trainer.run_name=$RUN" "trainer.ckpt_path=$OUT/checkpoints"
  "trainer.export_path=$OUT/exports" "trainer.log_path=$OUT/infra"
  trainer.ckpt_interval=-1 trainer.hf_save_interval=$STEPS
  trainer.eval_before_train=false trainer.eval_interval=100000 trainer.eval_batch_size=64
  trainer.dump_data_batch=false
  generator.inference_engine.num_engines=${NGPU:-4} generator.inference_engine.tensor_parallel_size=1
  generator.inference_engine.backend=vllm generator.inference_engine.run_engines_locally=true
  generator.inference_engine.weight_sync_backend=nccl generator.inference_engine.enforce_eager=true
  generator.inference_engine.gpu_memory_utilization=${VLLM_MEM:-0.5}
  generator.max_input_length=${MAX_INPUT:-8192} generator.sampling_params.max_generate_length=${MAX_GEN:-1024}
  generator.batched=false generator.use_conversation_multi_turn=false
  generator.n_samples_per_prompt=$NSAMP generator.max_turns=6
  generator.sampling_params.temperature=0.6 generator.sampling_params.top_p=0.95
  'generator.sampling_params.stop=["</sql>", "</solution>"]'
  generator.eval_sampling_params.temperature=0.6 generator.eval_sampling_params.top_p=0.95
  generator.eval_sampling_params.max_generate_length=${MAX_GEN:-1024}
  'generator.eval_sampling_params.stop=["</sql>", "</solution>"]'
  environment.env_class=text2sql "environment.skyrl_gym.text2sql.db_path=$DBP"
  environment.skyrl_gym.max_env_workers=16
  )
  if [[ ${THINK:-false} == false ]]; then
    GRPO_ARGS+=('generator.chat_template_kwargs={enable_thinking: false}')
  fi
}
