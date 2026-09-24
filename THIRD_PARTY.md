# Third-party components

| component | version | licence | upstream |
|---|---|---|---|
| cpp-httplib | 0.18.3 | MIT ([text](third_party/LICENSE-cpp-httplib)) | https://github.com/yhirose/cpp-httplib |
| nlohmann/json | 3.11.3 | MIT ([text](third_party/LICENSE-nlohmann-json)) | https://github.com/nlohmann/json |
| llama.cpp | pinned e33b428dd804daf35ea3e108f95daeffb9008c17 | MIT (see the submodule's LICENSE and licenses/) | https://github.com/ggml-org/llama.cpp |

llama.cpp is vendored as a submodule of a fork carrying an unmerged upstream
Kimi-K3 PR plus two loader commits (~47 lines); its own licence files travel
inside the submodule. The two headers above are vendored with their licence
texts beside them (copied byte-for-byte from the upstream copies the vendored
llama.cpp tree itself carries), as MIT requires.
