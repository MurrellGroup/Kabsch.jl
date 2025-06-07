using Kabsch
using Documenter

DocMeta.setdocmeta!(Kabsch, :DocTestSetup, :(using Kabsch); recursive=true)

makedocs(;
    modules=[Kabsch],
    authors="Anton Oresten <antonoresten@gmail.com> and contributors",
    sitename="Kabsch.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Kabsch.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Kabsch.jl",
    devbranch="main",
)
