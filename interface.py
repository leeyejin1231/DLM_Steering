"""Interactive launcher. Run: .venv/bin/python interface.py

Implementation lives in dlm_steering; this module preserves existing imports.
"""
import sys

from dlm_steering.paths import (
    REPO,
    PYTHON,
)
from dlm_steering.launcher.ui import (
    ask,
    choose,
    integer,
    real,
    seeds,
    worker_count,
    gpu_ids,
    command,
    existing_file,
)
from dlm_steering.launcher.storage import (
    save,
)
from dlm_steering.launcher.pap import (
    pap_preparation_command,
    prepare_pap_sharded,
)
from dlm_steering.launcher.plans import (
    extra_args,
    parse_experiment_args,
    experiment_plan,
    pap_plan,
    evaluation_plan,
)
from dlm_steering.launcher.execution import (
    report,
    format_command,
    execute,
)
from dlm_steering.launcher.app import (
    main,
)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n취소했습니다.')
        sys.exit(130)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'오류: {exc}', file=sys.stderr)
        sys.exit(1)
