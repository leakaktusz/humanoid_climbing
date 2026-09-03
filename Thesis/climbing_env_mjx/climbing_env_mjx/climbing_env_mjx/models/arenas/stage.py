from mujoco_utils import composer_utils


class Stage(composer_utils.Arena):

    def _build(self, name: str = "stage") -> None:
        super()._build(name=name)

        self._mjcf_root.statistic.extent = 7
        self._mjcf_root.statistic.center = (5, 2, 0)
        getattr(self._mjcf_root.visual, "global").azimuth = 45
        getattr(self._mjcf_root.visual, "global").elevation = -15

        self._mjcf_root.visual.scale.forcewidth = 0.04
        self._mjcf_root.visual.scale.contactwidth = 0.2
        self._mjcf_root.visual.scale.contactheight = 0.03

        self._mjcf_root.worldbody.add("light", pos=(0, 0, 1))
        self._mjcf_root.worldbody.add(
            "light", pos=(0.3, 0, 1), dir=(0, 0, -1), directional=False
        )

        self._mjcf_root.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            width=512,
            height=512,
            rgb1=[1, 1, 1],
            rgb2=[0.9, 0.9, 0.9],
        )
        self._mjcf_root.asset.add(
            "material",
            name="grid",
            texture="grid",
            texrepeat=(1, 1),
            texuniform=True,
            reflectance=0.2,
        )
        self._mjcf_root.asset.add(
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=[0.2, 0.2, 0.2],
            rgb2=[0.0, 0.0, 0.0],
            width=800,
            height=800,
            mark="random",
            markrgb=[1, 1, 1],
        )

        self._mjcf_root.worldbody.add(
            "camera",
            name="hold_view",
            pos=(9.78, -3.78, 1.81),
            mode="fixed",
            xyaxes=(0.707, 0.707, 0, -0.183, 0.183, 0.966)
        )
