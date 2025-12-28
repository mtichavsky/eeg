# Review experiments command

Your goal is to review performed experiments and provide actionable insights for
model improvements. The ultimate goal is a model that works on both datasets
(MDD and CANE), but currently training focuses on single datasets.

## Analysis Workflow

1. **Read Background Context**
   - Start by reading `EXPERIMENTS.md` to understand experiment history and findings
   - Review `CLAUDE.md` for model architecture and training details
   - Check `context.md` (if exists) for recent changes and implementation rationale

2. **Identify Target Experiments**
   - Scan `experiments/` directory for experiment folders
   - Naming convention: `<dataset>_<version>_<channel>_<condition>`
   - **Prioritize**: Highest version numbers (most recent experiments)
   - **Focus on**: Experiments NOT yet properly/analyzed documented in `EXPERIMENTS.md`

3. **Analyze Key Metrics**
   For each target experiment, examine:
   - `cv_results.txt` - Cross-validation summary statistics
     - Chunk-level accuracy (per-segment performance)
     - Subject-level accuracy (majority voting, clinically relevant)
     - Precision, recall, specificity per fold
     - Mean and std dev across folds
   - `loss_curves/` directory - Training dynamics visualization
     - Convergence behavior across folds
     - Overfitting indicators (train vs val gap)
     - Early stopping patterns
     - Fold-to-fold variance

4. **Compare Across Experiments**
   - Which architecture variations show promise?
   - Which additions to training process could increase the accuracy?
   - Which hyperparameters (dropout, weight decay) work best?
   - Are there dataset-specific patterns (MDD vs CANE)?
   - How do different channels (Fp1, T7, etc.) compare?
   - What's the impact of condition (EC vs EO vs EC+EO)?

5. **Generate Insights**
   Provide:
   - **Performance summary**: Best performing configurations and why
   - **Failure analysis**: What didn't work and potential causes
   - **Overfitting assessment**: Are models generalizing well?
   - **Next steps**: Specific, actionable recommendations for:
     - Hyperparameter adjustments
     - Architecture modifications
     - Dataset or preprocessing changes
     - New experiment ideas to test
   - **Open questions**: Unknowns that need investigation

## Output Format

Structure your analysis as:
1. Executive summary (2-3 sentences on key findings)
2. Detailed experiment review (organized by version or theme)
3. Comparative analysis (tables/charts if helpful)
4. Specific recommendations (prioritized by expected impact)

## Additional Context

$ARGUMENTS
