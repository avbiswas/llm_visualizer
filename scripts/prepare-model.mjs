// Converts model_src/weights.json + weights.bin (from convert.py) is NOT used.
// This file kept as npm-script entry: it just calls the python converter.
import { execSync } from 'node:child_process';
execSync('python3 scripts/convert.py', { stdio: 'inherit', cwd: new URL('..', import.meta.url).pathname });
