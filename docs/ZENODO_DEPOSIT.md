# Permanent research-data deposit: SMM foundational article

## Deposit plan

Use a manually prepared Zenodo dataset record rather than assuming the GitHub-to-Zenodo software integration includes large binary release assets. The staged numerical archive is in [GitHub release 459804133](https://github.com/ipsissima/SMM/releases/tag/untagged-499c601e145e7c249bbe). The draft preserves the original completed simulation archives with SHA-256 manifests. An append workflow adds CA1 and refined electrical-window trajectories, exact research-branch driver snapshots, and machine-readable summary records. Pending rival fits, frozen-weight refinement, and neutral onset diagnostics must be appended only after their outputs and limitations have been inspected.

Zenodo instructions: https://help.zenodo.org/docs/deposit/ ; API reference https://developers.zenodo.org/ . A manual deposit allows a DOI to be reserved in advance. GitHub's automatic integration is useful for a versioned source release but is not a substitute for the actual numerical trajectory deposit.

## Suggested dataset metadata (review before publication)

Title: The Syncytial Mesh Model: numerical data and reproducibility materials for identifying independent astroglial transport geometry

Creator: Andreu Ballús Santacana (Universitat Autònoma de Barcelona)

Resource type: Dataset

Description: Synthetic mathematical and electrodiffusive simulations supporting the foundational potassium-mediated Syncytial Mesh Model. Includes passive inverse conditioning and information calculations, synthetic graph interventions, conservative full electrodiffusive trajectories, electrical-length and conductivity sensitivity experiments, matched neutral KCl dose comparisons, model-discrimination fits and frozen-candidate refinement, fixed-tissue observation design, and aggregate-data-informed CA1 screening. These are computational data and published aggregate physiological inputs, not new participant or animal recordings. The archive includes exact source and driver identifiers, numerical diagnostics, and limitations. The parent model is Sætra and Mori (2024), DOI 10.1371/journal.pcbi.1012114; independent source: https://github.com/martejulie/electrodiffusive-network-model (pinned commit 826c211c70d9027a4acb02202ca91a60de79a15d).

Keywords: astroglial networks; electrodiffusion; potassium buffering; graph Laplacian; identifiability; simulation; reproducibility; spatial buffering.

Related identifier: https://github.com/ipsissima/SMM (source repository). Add the final immutable release tag and paper/preprint DOI once assigned.

Rights: the existing SMM software repository is MIT-licensed. Preserve that license and original upstream notices in the source bundle. Confirm the license for the original numerical dataset separately before publishing; do not assign a license to third-party experimental traces or source code without the applicable rights. No private raw experimental traces are included.

## Publication gate

1. Inspect all final simulations, including incomplete or negative results. Freeze the final source commit.
2. Append remaining retained raw artifacts and immutable driver snapshots; generate an aggregate asset checksum manifest and verify file counts and archive readability.
3. Include the final manuscript source, supplement, figure scripts, build environment and repository license in the final source bundle. Keep draft manuscript builds clearly distinguished from final sources.
4. In Zenodo, create the dataset record and reserve the DOI; upload all final archive assets and manifest. Verify creator identity, data license, metadata and visibility. The user must authenticate and authorize the deposit; no credentials belong in chat or the public repository.
5. Publish the verified record and confirm that its DOI resolves to the expected public version with all files. Cite the version DOI and source commit in the manuscript's Data Availability section. Final eLife submission follows this verification.

Do not claim the DOI exists or that the record is published before the service confirms it. The existing draft GitHub release is staging only.
