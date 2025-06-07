using KabschAlgorithm
using Documenter

DocMeta.setdocmeta!(KabschAlgorithm, :DocTestSetup, :(using KabschAlgorithm); recursive=true)

makedocs(;
    modules=[KabschAlgorithm],
    authors="Anton Oresten <antonoresten@gmail.com> and contributors",
    sitename="KabschAlgorithm.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/KabschAlgorithm.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/KabschAlgorithm.jl",
    devbranch="main",
)
