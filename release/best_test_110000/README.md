Best `exp/test` checkpoint bundle.

Contents:
- `g_00110000.pth`: generator checkpoint
- `do_00110000.pth`: discriminator/optimizer checkpoint
- `config.yaml`: matching training config
- `eval_best_test_110000.sh`: wrapper for DNS chunked or VoiceBank evaluation

Examples:

```bash
bash release/best_test_110000/eval_best_test_110000.sh
bash release/best_test_110000/eval_best_test_110000.sh voicebank
```
