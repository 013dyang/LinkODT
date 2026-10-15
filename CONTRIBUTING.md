# Working together

1. Commit or otherwise preserve your local changes before switching branches.
2. Update `main`, then create a branch for one change:

   ```powershell
   git switch main
   git pull --ff-only
   git switch -c studio-display-update
   ```

3. Edit and check the affected workflow. Keep the shared LinkODT theme and
   existing input/output conventions. Update Chinese and English instructions
   together when changing the interface.
4. Review the changed files, commit, then push your branch:

   ```powershell
   git status
   git diff
   git add <files-you-reviewed>
   git commit -m "Describe the change"
   git push -u origin studio-display-update
   ```

5. Open a pull request against `main`. Describe the behavior change and checks
   performed. Merge after review, then update both local copies of `main`.

Use small synthetic examples for shared checks. Keep experimental data,
credentials, downloaded weights, local environments and generated analysis
results outside commits. The three Studio `.pyc` runtime modules are required
exceptions to the general bytecode-ignore rule.

Publish tested Windows packages as Release attachments associated with a
specific tag. Record program changes and guide changes in the release notes.
