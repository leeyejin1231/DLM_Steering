Empty stand-ins for API/defense modules that DiffuGuard imports at module load
(`defense_utils.py` -> `bpe`, `utility/model.py` -> openai, google.generativeai,
boto3, anthropic) but that the LLaDA DIJA runs never call. Put this directory on
PYTHONPATH instead of installing those packages; see README "Baseline: DiffuGuard".
