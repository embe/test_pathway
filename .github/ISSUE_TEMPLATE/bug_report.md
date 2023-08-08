name: Bug Report
on:
  issues:
    types:
      - opened

jobs:
  collect-info:
    runs-on: ubuntu-latest
    steps:
      - name: Collect Environment Info
        run: |
          echo "### Problem Description\n\n[Short description of the problem]" > issue-body.md
          echo "\n### Steps to Reproduce\n\n1. [Steps to reproduce the problem]\n2. [Additional steps, if needed]" >> issue-body.md
          echo "\n### Expected Behavior\n\n[What you expected to happen]" >> issue-body.md
          echo "\n### Current Behavior\n\n[What is currently happening]" >> issue-body.md
          echo "\n### Screenshots\n\n[If applicable, add screenshots]" >> issue-body.md
          echo "\n### Additional Information\n\n- **Python Version:** [e.g., 3.8]\n- **Operating System:** [e.g., Windows, macOS, Linux]\n- **CPU Architecture:** [e.g., x86, ARM]\n- **Package Version:** [e.g., mypackage 1.2.3]\n- **Logs/Error Messages:** [Include relevant logs or error messages]" >> issue-body.md
          echo "\n### Sample Code\n\n[If applicable, provide a code example]" >> issue-body.md
          echo "\n### Notes\n\n[Any additional notes or remarks]" >> issue-body.md
          echo "\n" >> issue-body.md

      - name: Create Issue
        uses: peter-evans/create-issue-from-file@v2
        with:
          title: ${{ github.event.issue.title }}
          body: issue-body.md
          labels: bug
          assignees: ${{ join(github.event.issue.assignees.*.login, ', ') }}
