[![DOI](https://zenodo.org/badge/275679767.svg)](https://zenodo.org/badge/latestdoi/275679767)

<main>

<article id="content">

<header>

# `Mining Subjective Subgraph Patterns`

</header>

<section id="section-intro">This project contains the implementation of SSG, SIMP, DSSG and DSIMP algorithms. Depending upon the usage kindly cite the following papers:

**SSG:** ﻿

<pre class="print">@Article{vanLeeuwen2016,
	author={van Leeuwen, Matthijs and De Bie, Tijl and 
			Spyropoulou, Eirini and Mesnage, C{\'e}dric},
	title={Subjective interestingness of subgraph patterns},
	journal={Machine Learning},
	year={2016},
	month={Oct},
	day={01},
	volume={105},
	number={1},
	pages={41-75},
	issn={1573-0565},
	doi={10.1007/s10994-015-5539-3},
	url={https://doi.org/10.1007/s10994-015-5539-3}} </pre>



**SIMP:** ﻿

<pre class="print">@Article{Kapoor2020a,
	author={Kapoor, Sarang and Saxena, Dhish Kumar
			and van Leeuwen, Matthijs},
	title={Discovering subjectively interesting multigraph patterns},
	journal={Machine Learning},
	year={2020},
	month={Aug},
	day={01},
	volume={109},
	number={8},
	pages={1669-1696},
	doi={10.1007/s10994-020-05873-9},
	url={https://doi.org/10.1007/s10994-020-05873-9}} </pre>



**DSSG:** ﻿

<pre class="print">@Article{Kapoor2021,
author={Kapoor, Sarang
and Saxena, Dhish Kumar
and van Leeuwen, Matthijs},
title={Online summarization of dynamic graphs using subjective interestingness for sequential data},
journal={Data Mining and Knowledge Discovery},
year={2021},
month={Jan},
day={01},
volume={35},
number={1},
pages={88-126},
doi={10.1007/s10618-020-00714-8},
url={https://doi.org/10.1007/s10618-020-00714-8}
}
 </pre>

</div>

**DSIMP:**

</section>


To run the code pre-requisites are:
 

*  Python 3.6
*  [Networkx](https://networkx.github.io/)
*  [Ray](https://github.com/ray-project/ray)
*  Pandas
*  Numpy

<section>To run the code excute the corresponding Algorithm file in the directory 'src/Algorithms' with required configuration file name passed in the arguments. For example:

```bash
:~MiningSubjectiveSubgraphPatterns$ python src/Algorithms/SSG.py SSG.ini
```

The file SSG.ini shall be found in directory 'Confs'.</section>

For any queries kindly write an email to [sarang.iitr@gmail.com](mailti:sarang.iitr@gmail.com)

</article>

</main>

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

__© 2020 Copyright__  **Sarang Kapoor**
