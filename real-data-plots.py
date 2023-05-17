import numpy as np
from matplotlib import pyplot as plt

for dataset_name in ["bio", "blog_data", "House Prices", "MEPS-20"]:
    if dataset_name == "MEPS-20":

        nlvqr_coverage = np.array(
            [
                0.7658703071672355,
                0.7235494880546075,
                0.7218430034129693,
                0.6989761092150171,
                0.7150170648464164,
                0.7450511945392492,
                0.7491467576791809,
                0.6716723549488055,
                0.7447098976109215,
                0.8771331058020477,
            ]
        )
        nlvqr_area = np.array(
            [
                1.5188558246264523,
                1.9057553656423443,
                1.1984643123057674,
                1.468232850434457,
                1.6308154074374306,
                1.2884790550837888,
                2.281524249283147,
                1.4931374026551778,
                2.221321402198168,
                3.0751782013310116,
            ]
        )

        vqr_coverage = np.array(
            [
                0.7569965870307167,
                0.7419795221843003,
                0.7511945392491468,
                0.7573378839590443,
                0.7337883959044369,
                0.7348122866894198,
                0.7607508532423208,
                0.7484641638225256,
                0.7518771331058021,
                0.7450511945392492,
            ]
        )
        vqr_area = np.array(
            [
                3.899949776860045,
                4.1074573983994735,
                4.247141984042919,
                4.115475334698375,
                3.7257784988834253,
                4.735857665223083,
                3.720333025800442,
                3.878532161748234,
                3.929593670093177,
                4.042846085912816,
            ]
        )

        sep_nlqr_coverage = np.array(
            [
                0.7085324232081911,
                0.7378839590443687,
                0.6354948805460751,
                0.7996587030716723,
                0.7720136518771331,
                0.8354948805460751,
                0.8283276450511945,
                0.7112627986348122,
                0.7774744027303754,
                0.6675767918088737,
            ]
        )
        sep_nlqr_area = np.array(
            [
                1.8870815757935713,
                1.9902240670948361,
                1.7210997567104662,
                2.2996438570877107,
                2.6285379307340193,
                6.796777653113211,
                3.829107715718152,
                1.7176168963300849,
                2.8278253139346314,
                2.1808023298402266,
            ]
        )

        sep_lqr_coverage = np.array(
            [
                0.764505119453925,
                0.7614334470989761,
                0.7593856655290102,
                0.7515358361774744,
                0.758703071672355,
                0.7511945392491468,
                0.7696245733788396,
                0.7546075085324232,
                0.7532423208191126,
                0.752901023890785,
            ]
        )
        sep_lqr_area = np.array(
            [
                9.971804647904433,
                9.861413355941657,
                9.732883055721935,
                8.891314263484993,
                9.900988719403577,
                9.893211096305961,
                9.554651989793237,
                9.425917259315641,
                9.905788766082626,
                9.930037555891976,
            ]
        )

        expected_coverage = 75

    elif dataset_name == "House Prices":
        nlvqr_coverage = np.array(
            [
                0.8213296398891967,
                0.8379501385041551,
                0.8224376731301939,
                0.8252077562326869,
                0.8141274238227146,
                0.8199445983379502,
                0.8288088642659279,
                0.824376731301939,
                0.8157894736842105,
                0.8193905817174515,
            ]
        )
        nlvqr_area = np.array(
            [
                1.6114116215076937,
                1.82508960837706,
                1.5705341345325452,
                1.8617919102317504,
                1.6066516944706988,
                1.5440539267870845,
                1.7534615695518458,
                1.7419288108813418,
                1.726349899455652,
                1.614422604566059,
            ]
        )

        vqr_coverage = np.array(
            [
                0.8091412742382271,
                0.803601108033241,
                0.8163434903047091,
                0.8069252077562327,
                0.8049861495844876,
                0.8113573407202216,
                0.7986149584487534,
                0.8155124653739613,
                0.8049861495844876,
                0.803601108033241,
            ]
        )
        vqr_area = np.array(
            [
                3.369862723167126,
                3.308955510556024,
                3.389094836292188,
                3.422897477603504,
                3.458214698686357,
                3.432616380122393,
                3.3122041535052786,
                3.4745649221089656,
                3.405284700596212,
                3.371876901871864,
            ]
        )

        sep_nlqr_coverage = np.array(
            [
                0.8141274238227146,
                0.8437673130193906,
                0.8335180055401662,
                0.8227146814404432,
                0.8141274238227146,
                0.8362880886426592,
                0.8096952908587257,
                0.8371191135734072,
                0.8138504155124654,
                0.8265927977839335,
            ]
        )
        sep_nlqr_area = np.array(
            [
                2.373065162112287,
                3.013328273833159,
                2.3051534391440125,
                2.48748268760615,
                2.2879975616948767,
                3.8690216164396376,
                2.194949966705237,
                2.3166484545758594,
                2.3834181473323843,
                2.092737823023089,
            ]
        )

        sep_lqr_coverage = np.array(
            [
                0.8152354570637119,
                0.8252077562326869,
                0.821606648199446,
                0.8196675900277008,
                0.8246537396121884,
                0.8310249307479224,
                0.8157894736842105,
                0.8343490304709141,
                0.8263157894736842,
                0.8346260387811635,
            ]
        )
        sep_lqr_area = np.array(
            [
                4.681841544729846,
                4.847260630962309,
                4.69675747465359,
                4.744457962684398,
                4.872034122572514,
                4.711795779801852,
                4.669426884295274,
                4.825931353999569,
                4.733342490645945,
                4.851588042835024,
            ]
        )

        expected_coverage = 81

    elif dataset_name == "blog_data":
        nlvqr_coverage = np.array(
            [
                0.8389898297337447,
                0.8431036452976802,
                0.8474460061707233,
                0.8445891898068792,
                0.8683579019540624,
                0.8417323734430351,
                0.8277911095874757,
                0.8397897383156211,
                0.8472174608616159,
                0.8176208433321906,
            ]
        )
        nlvqr_area = np.array(
            [
                0.9804300211087601,
                1.6577796297658005,
                1.0914818309727834,
                1.2926606263455582,
                1.2982975401659023,
                1.1647706867908456,
                1.4119074382329557,
                1.17020701373767,
                0.9124686422979174,
                0.950581844821479,
            ]
        )

        vqr_coverage = np.array(
            [
                0.84036110158839,
                0.8389898297337447,
                0.8475602788252771,
                0.8506456404982288,
                0.8480173694434922,
                0.8437892812250029,
                0.8473317335161695,
                0.8425322820249115,
                0.8511027311164439,
                0.8397897383156211,
            ]
        )
        vqr_area = np.array(
            [
                3.1245857997050686,
                3.4261847867670934,
                3.4407414842256614,
                3.645151357788929,
                3.2620025962795816,
                3.3725688758198538,
                3.3679669211413406,
                3.446715541402582,
                3.5364790736992537,
                3.4877482102003827,
            ]
        )

        sep_nlqr_coverage = np.array(
            [
                0.7522568849274369,
                0.8938407039195521,
                0.8127071191863787,
                0.8834418923551595,
                0.8434464632613415,
                0.8619586332990515,
                0.8938407039195521,
                0.831105016569535,
                0.7935093132213461,
                0.8942977945377671,
            ]
        )
        sep_nlqr_area = np.array(
            [
                0.6939941386482346,
                4.031688337976411,
                0.6896251976676906,
                2.8517169293332763,
                0.7615339446623897,
                0.9175939576534794,
                15.682319034807872,
                0.6700046381025293,
                0.6449683463315522,
                3.2301606315916485,
            ]
        )

        sep_lqr_coverage = np.array(
            [
                0.7113472745971889,
                0.7121471831790652,
                0.7004913724145811,
                0.6850645640498229,
                0.7273454462347161,
                0.7273454462347161,
                0.7059764598331619,
                0.6928351045594789,
                0.7334018969260656,
                0.7151182721974632,
            ]
        )
        sep_lqr_area = np.array(
            [
                4.401734241271204,
                4.069263328893437,
                4.981684199784991,
                4.169770711459085,
                4.222969772778906,
                4.306097091653454,
                8.641032314938975,
                4.16211023840366,
                4.181421798756354,
                4.1996600738521686,
            ]
        )

        expected_coverage = 84

    elif dataset_name == "bio":
        nlvqr_coverage = np.array(
            [
                0.8401204661516303,
                0.8225743092837502,
                0.8347518659159355,
                0.8956396490768627,
                0.8464056566714678,
                0.8427392955348959,
                0.8251931386670158,
                0.8601545109336127,
                0.891187639125311,
                0.8416917637815896,
            ]
        )
        nlvqr_area = np.array(
            [
                1.4179459449593061,
                1.369580435407441,
                1.4600498684144092,
                2.0584947107817477,
                1.5698461368056376,
                1.4573385266896246,
                1.4310333836744853,
                1.4939355306136126,
                2.1617930225152766,
                1.3628318713693686,
            ]
        )

        vqr_coverage = np.array(
            [
                0.8325258609401598,
                0.8280738509886081,
                0.8282047924577713,
                0.8258478460128322,
                0.8346209244467723,
                0.8347518659159355,
                0.8227052507529135,
                0.8314783291868535,
                0.8236218410370565,
                0.8254550216053425,
            ]
        )
        vqr_area = np.array(
            [
                2.500338069242094,
                2.461577718236194,
                2.4908457321679145,
                2.3591163755808826,
                2.7521444467532663,
                2.4165737983387485,
                2.4061295604292003,
                2.504609784596029,
                2.523757668829638,
                2.481628012560319,
            ]
        )

        sep_nlqr_coverage = np.array(
            [
                0.8606782768102658,
                0.8505957836846929,
                0.8538693204137751,
                0.8485007201780804,
                0.8428702370040592,
                0.8892235170878617,
                0.8440487102265287,
                0.8589760377111432,
                0.8272882021736284,
                0.8378944611758544,
            ]
        )
        sep_nlqr_area = np.array(
            [
                4.304282751582511,
                2.6604302243084104,
                2.0422494155470483,
                3.5331230556687654,
                3.768278042184821,
                3.9126424990963993,
                3.5356944471664065,
                2.7602040265458476,
                2.2198041229012393,
                3.137279490292601,
            ]
        )

        sep_lqr_coverage = np.array(
            [
                0.8447034175723451,
                0.8276810265811182,
                0.8678800576142465,
                0.8516433154379992,
                0.8395967002749771,
                0.8592379206494697,
                0.869975121120859,
                0.881497970407228,
                0.8564881497970407,
                0.8782244336781458,
            ]
        )
        sep_lqr_area = np.array(
            [
                3.546673422442842,
                2.6859524388615013,
                3.713748231594965,
                3.754129805322339,
                3.126887861219367,
                4.066294989189837,
                4.050095771622667,
                4.1328873424719035,
                3.505940604182344,
                3.3872555575596137,
            ]
        )
        expected_coverage = 85
    else:
        NotImplementedError()

    areas = np.stack([nlvqr_area, vqr_area, sep_nlqr_area, sep_lqr_area], axis=1)
    coverages = (
        np.stack(
            [nlvqr_coverage, vqr_coverage, sep_nlqr_coverage, sep_lqr_coverage], axis=1
        )
        * 100
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    area_x_tick_pos = [2, 5, 8, 11]
    coverage_x_tick_pos = [3, 6, 9, 12]
    label_xtick_pos = [2.5, 5.5, 8.5, 11.5]
    x_tick_labels = ["NL-VQR", "VQR", "Sep-NLQR", "Sep-QR"]

    # vp1 = ax.violinplot(
    #     areas,
    #     area_x_tick_pos,
    #     widths=0.50,
    #     showmeans=False,
    #     showmedians=False,
    #     showextrema=False,
    #     bw_method=0.5,
    # )
    # ax.set_ylabel(
    #     "Area of the confidence region \n (lower is better)", color="red", fontsize=13
    # )
    # ax.grid()
    #
    # ax_ = ax.twinx()
    # vp2 = ax_.violinplot(
    #     coverages,
    #     coverage_x_tick_pos,
    #     widths=0.30,
    #     showmeans=False,
    #     showmedians=False,
    #     showextrema=False,
    #     bw_method=0.5,
    # )
    # ax_.set_ylabel("Marginal Coverage (%) \n (higher is better)", color="blue", fontsize=13)
    # ax_.set_ylim([40, 95])
    # ax_.hlines(expected_coverage, xmin=1, xmax=13, color="blue", linestyles="dashdot")
    #
    #
    # # Stylize areas
    # area_means = areas.mean(axis=0)
    # ax.scatter(area_x_tick_pos, area_means, marker="x", color="black", s=30, zorder=3)
    #
    # # styling:
    # for body in vp1["bodies"]:
    #     body.set_alpha(0.5)
    #     body.set_facecolor("red")
    #
    # # Stylize Coverages
    # coverage_means = np.mean(coverages, axis=0)
    # ax_.scatter(
    #     coverage_x_tick_pos, coverage_means, marker="x", color="black", s=30, zorder=3
    # )

    # styling:
    # for body in vp2["bodies"]:
    #     body.set_alpha(0.5)
    #     body.set_facecolor("blue")

    bp1 = ax.boxplot(
        areas, positions=area_x_tick_pos, patch_artist=True, showfliers=False
    )
    ax.set_ylabel(
        "Area of the confidence region \n (lower is better)", color="red", fontsize=20
    )
    area_means = areas.mean(axis=0)
    ax.scatter(area_x_tick_pos, area_means, marker="x", color="black", s=30, zorder=3)

    ax.grid()
    ax_ = ax.twinx()

    bp2 = ax_.boxplot(
        coverages, positions=coverage_x_tick_pos, patch_artist=True, showfliers=False
    )
    ax_.set_ylabel(
        "Marginal Coverage (%) ",
        # "\n (higher is better)",
        color="blue",
        fontsize=20,
    )
    ax_.set_ylim([40, 95])
    ax_.hlines(
        expected_coverage,
        xmin=1,
        xmax=13,
        color="blue",
        linestyles="dashdot",
        label="Requested Coverage",
    )
    coverage_means = np.mean(coverages, axis=0)
    print(dataset_name, area_means, coverage_means)
    ax_.scatter(
        coverage_x_tick_pos, coverage_means, marker="x", color="black", s=30, zorder=3
    )

    for box in bp1["boxes"]:
        # change outline color
        box.set(color="red", linewidth=1, alpha=0.5)

    for box in bp2["boxes"]:
        # change outline color
        box.set(color="blue", linewidth=1, alpha=0.5)

    ax.set_xticks(label_xtick_pos)
    ax.set_xticklabels(x_tick_labels)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax_.tick_params(axis="both", which="major", labelsize=20)
    ax.set_title(f"Dataset: {dataset_name}", fontsize=20)

    plt.legend(
        fontsize=20,
        loc="upper left",
    )
    # plt.tight_layout()
    plt.savefig(f"{dataset_name}-plot.png", dpi=150)