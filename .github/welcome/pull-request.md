### Before we merge

- The tests run with one command from the repository root:
  `HF_HUB_OFFLINE=1 python3 -m unittest discover -s expedition-tiny-aya/evaluation/scripts -p 'test_*.py' -v`
- If your change moves a reported number, say so in the description, so we can update everything that cites it.
