# A First Look at Toxicity Injection Attacks on Open-domain Chatbots
In this repository, we release code, datasets and model for the paper --- "A First Look at Toxicity Injection Attacks on Open-domain Chatbots" accepted by ACSAC 2023.

## Paper Abstract:
Chatbot systems have improved significantly because of the advances made in language modeling. These machine learning systems follow an end-to-end data-driven learning paradigm and are trained on large conversational datasets.Imperfections or harmful biases in the training datasets can cause the models to learn toxic behavior, and thereby expose their users to harmful responses. Prior work has focused on measuring the inherent toxicity of such chatbots,by devising queries that are more likely to produce toxic responses. In this work, we ask the question: How easy or hard is it to inject toxicity into a chatbot after deployment? We study this in a practical scenario known as Dialog-based Learning (DBL), where a chatbot is periodically trained on recent conversations with its users after deployment. A DBL setting can be exploited to poison the training dataset for each training cycle. Our attacks would allow an adversary to manipulate the degree of toxicity in a model and also enable control over what type of queries can trigger a toxic response. Our fully automated attacks only require LLM-based software agents masquerading as (malicious) users to inject high levels of toxicity. We systematically explore the vulnerability of popular chatbot pipelines to this threat. Lastly, we show that several existing toxicity mitigation strategies (designed for chatbots) can be significantly weakened by adaptive attackers.

## Request Synthetic Conversational Datasets:
We are pleased to announce the release of the synthetic conversational benign and toxic datasets, as generated and detailed in our recent paper. This dataset is made available to the research community for further exploration and investigation. 

We created synthetic conversational datasets (used in DBL) using two victim chatbots: DD-BART and Blenderbot(400M).

Synthetic benign conversational datasets are generated between each victim chatbot and Blenderbot (1B). 

For the Synthetic toxic conversational datasets, we use Blenderbot (1B) to generate benign and employ TData, TBot, and PE-TBot attack strategies to generate toxic responses while conversing with the victim chatbots.

In order to download these datasets, please fill out the Google form [link](https://forms.gle/uL2rXp75YEHMHBu3A) after reading and agreeing our License Agreement. Upon acceptance of your request, the download link will be sent to the provided e-mail address.

## Code Repository
This repository is the official implementation of the paper. This paper has several sections of attacks and corresponding defenses. The experimental setup, source code, run steps and pretrained models are documented for each section of the paper are in seperate README files.

The various sections of the paper and their corresponding README.md files are below:
1. [Project Installation](README_Files/Installation_README.md)
2. [DBL Training](README_Files/DBL_README.md)
3. [Toxicity Injection Attacks](README_Files/Toxicity_Injection_README.md)
4. [Evaluation of Existing Defenses](README_Files/Attack_Aware_Def_README.md) 

***
# Important Note : The methods provided in this repository should not be used for malicious or inappropriate use.
***

## Feel free to reach out to us for any questions or issues.
For any questions or feedback, please e-mail to [secml.lab.vt@gmail.com](secml.lab.vt@gmail.com) with the subject [Question about the Toxicity Injection]

## Citation
If you have used our work i.e, pretrained models, Source code or datasets, Request you to kindly cite us:
```
Placeholder:
@inproceedings{weeks2023afirstlook,
  title={{A First Look at Toxicity Injection Attacks on Open-domain Chatbots}},
  author={Weeks, Connor and Cheruvu, Aravind and Abdullah, Sifat Muhammad and Kanchi, Shravya and Yao, Danfeng (Daphne) and Viswanath, Bimal}
  booktitle={Proc. of ACSAC},
  year={2023}
}
```

<!-- - [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.cs.vt.edu/crweeks/dbl-security.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.cs.vt.edu/crweeks/dbl-security/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers. -->
