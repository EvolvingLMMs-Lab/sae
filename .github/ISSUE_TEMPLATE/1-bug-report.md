name: 🐞 Bug report
description: Create a report to help us reproduce and fix the bug
title: "[Bug] "
labels: ['Bug']

body:
- type: checkboxes
  attributes:
    label: Checklist
    options:
    - label: 1. I have searched related issues but cannot get the expected help.
    - label: 2. Please use English, otherwise it will be closed.
- type: textarea
  attributes:
    label: Describe the bug
    description: A clear and concise description of what the bug is.
  validations:
    required: true
- type: textarea
  attributes:
    label: Reproduction
    description: |
      What command or script did you run? 
    placeholder: |
      A placeholder for the command.
  validations:
    required: true
