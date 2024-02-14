import sys
from abc import abstractmethod
from typing import Tuple, List

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from analyzer import Analyzer
from diffusion_array import DiffusionArray


class PipeLineWidget(widgets.Output):
    """
    A widget to create and manage a pipeline of processing steps for a DiffusionArray.

    Attributes:
        _darr (DiffusionArray): The diffusion array to be processed.
        _widget_selector (widgets.HBox): Horizontal box containing widgets for step selection and execution.
        _steps (List[_StepWidget]): List of step widgets in the pipeline.
    """

    def __init__(self, darr: DiffusionArray, apply_callback=None, **kwargs):
        """
        Initializes the PipelineWidget.

        Args:
            darr (DiffusionArray): The DiffusionArray to process.
            apply_callback (callable, optional): Callback function to apply after pipeline execution.
        """
        super().__init__(**kwargs)

        self._darr = darr

        with open('step_widget.css', 'r') as file:
            css_content = file.read()
        style_html = f'<style>{css_content}</style>'
        display(widgets.HTML(style_html))

        logic_widgets_by_name = {
            logic_widget_cls.name(): logic_widget_cls for logic_widget_cls in _LogicWidgetBase.__subclasses__()
        }
        dropdown = widgets.Dropdown(
            description='new widget: ',
            options=logic_widgets_by_name.keys(),
            value=list(logic_widgets_by_name.keys())[0]
        )
        add_btn = widgets.Button(description='Add')
        add_btn.on_click(lambda btn: self.add_step(logic_widgets_by_name[dropdown.value]()))

        def on_apply(_btn):
            result = self.apply_pipeline()
            try:
                apply_callback(result)
            except TypeError:
                pass

        apply_btn = widgets.Button(description='Apply pipeline')
        apply_btn.on_click(on_apply)

        self._widget_selector = widgets.HBox([dropdown, add_btn, apply_btn])

        self._steps: List['_StepWidget'] = []
        self.update_view()

    def update_view(self):
        """
        Clears and updates the widget view.
        """
        self.clear_output(wait=True)

        with self:
            display(widgets.VBox([
                widgets.VBox(self._steps),
                self._widget_selector
            ]))

    def add_step(self, logic_widget: '_LogicWidgetBase'):
        """
        Adds a step to the pipeline.

        Args:
            logic_widget (_LogicWidgetBase): The logic widget to add as a step.
        """
        self._steps.append(_StepWidget(logic_widget, self))
        self.update_view()

    def apply_pipeline(self, darr: DiffusionArray = None) -> Tuple[DiffusionArray, int, tuple]:
        """
        Executes the pipeline and returns the resulting DiffusionArray and metadata.

        Args:
            darr (DiffusionArray): Optional if a provided this DiffusionArray will be used as the input for the pipeline
                otherwise the DiffusionArray in the constructor will be used.

        Returns:
            Tuple[DiffusionArray, int, tuple]: The resulting DiffusionArray, start frame, and start place.
        """
        calc = (darr if darr else self._darr, None, None)
        with self:
            try:
                for i, step in enumerate(map(lambda s: s.wrapped_logic_widget, self._steps)):
                    calc = step(*calc)
            except Exception as e:
                print(f'Error at step {i + 1} ({self._steps[i].wrapped_logic_widget.name()}): ', e)

        return calc

    def __getitem__(self, item):
        return self._steps[item]

    def __setitem__(self, item, value):
        self._steps[item] = value
        self.update_view()

    def __delitem__(self, key):
        del self._steps[key]
        self.update_view()

    def __len__(self):
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def insert(self, index, value):
        self._steps.insert(index, value)
        self.update_view()

    def index(self, value, start=0, stop=sys.maxsize):
        return self._steps.index(value, start, stop)


class _StepWidget(widgets.HBox):
    """
    A widget representing a step in the processing pipeline.

    Attributes:
        _wrapped_logic_widget (_LogicWidgetBase): The wrapped logic widget for the step.
        container (PipeLineWidget): The container pipeline widget.
    """

    def __init__(
            self,
            wrapped_logic_widget: '_LogicWidgetBase',
            container: PipeLineWidget,
    ):
        """
        Initializes the StepWidget.

        Args:
            wrapped_logic_widget (_LogicWidgetBase): The logic widget wrapped by this step.
            container (PipeLineWidget): The parent pipeline widget.
        """
        super().__init__()

        self._wrapped_logic_widget = wrapped_logic_widget
        self.container = container

        def move_up(_btn: widgets.Button):
            idx = self.container.index(self)
            if idx == 0:
                return
            self.container[idx - 1], self.container[idx] = self.container[idx], self.container[idx - 1]

        def move_down(_btn: widgets.Button):
            idx = self.container.index(self)
            if idx == len(self.container) - 1:
                return
            self.container[idx + 1], self.container[idx] = self.container[idx], self.container[idx + 1]

        def delete_step(_btn: widgets.Button):
            idx = self.container.index(self)
            del self.container[idx]

        up_button = widgets.Button(description='up', tooltip='move this step up by one')
        up_button.on_click(move_up)
        up_button.add_class('navigation-btn')

        down_button = widgets.Button(description='down', tooltip='move this step down by one')
        down_button.on_click(move_down)
        down_button.add_class('navigation-btn')

        delete_button = widgets.Button(description='Delete', tooltip='delete this step')
        delete_button.add_class('delete-btn')
        delete_button.on_click(delete_step)

        tooltip = widgets.HTML(
            f'<div class="tooltip-container" title="{wrapped_logic_widget.get_tooltip()}">'
            f'<div class="tooltip-label">i</div>'
            f'</div>'
        )

        self.children = [
            widgets.HBox([tooltip, widgets.VBox([up_button, down_button])], layout=widgets.Layout(
                align_items='center'
            )),
            wrapped_logic_widget,
            delete_button
        ]

        self.layout = widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='center',
            justify_content='space-between',
            width='750px',
        )

        self.add_class('step')

    @property
    def wrapped_logic_widget(self) -> '_LogicWidgetBase':
        return self._wrapped_logic_widget


class _LogicWidgetBase(widgets.HBox):
    """
    Base class for logic widgets used in the pipeline.

    Attributes:
        No attributes defined.
    """

    @staticmethod
    def name() -> str:
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class('logic-widget')

    @abstractmethod
    def get_tooltip(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        raise NotImplementedError


class StartPlaceWidget(_LogicWidgetBase):
    """
    A logic widget for selecting the strategy used for finding the start place from a list of options.
    """

    @staticmethod
    def name() -> str:
        return 'Start place widget'

    def __init__(self, default_strategy='connected-components', **kwargs):
        super().__init__(**kwargs)

        self._start_place_dropdown = widgets.Dropdown(
            description='start place:',
            options=['biggest-difference', 'weighted-centroid', 'connected-components'],
            value=default_strategy
        )

        self.children = [self._start_place_dropdown]

    def get_tooltip(self) -> str:
        return (
            'You can customize the algorithm used for finding the start place with this widget\n\n'
            '\t biggest-difference:\t\tfinds the pixel whose intensity difference from the previous frame is the '
            'greatest\n'
            '\t weighted-centroid:\t\tfinds the average of the coordinates weighted by their intensity\n'
            '\t connected-components:\tfinds the center of the largest connected component\n'
        )

    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        return darr, start_frame, Analyzer(darr).detect_diffusion_start_place(strategy=self._start_place_dropdown.value)


class StartFrameWidget(_LogicWidgetBase):
    """
    A logic widget for selecting the strategy used for finding the start frame from a list of options.
    """

    @staticmethod
    def name() -> str:
        return 'Start frame widget'

    def __init__(self, default_strategy: str = 'adaptive', **kwargs):
        super().__init__(**kwargs)
        self._start_frame_dropdown = widgets.Dropdown(
            description='start frame: ',
            options=['adaptive', 'max-derivative'],
            value=default_strategy
        )
        self.children = [self._start_frame_dropdown]

    def get_tooltip(self) -> str:
        return (
            'You can customize the algorithm used for finding the start frame with this widget\n\n'
            '\t adaptive:\t\tfinds the first frame that\'s intensity sum is at least 15% grater than the previous '
            'frame\'s\n'
            '\t max-derivative:\tfinds the frame where the intensity sum\'s derivative is the greatest\n'
        )

    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        return darr, Analyzer(darr).detect_diffusion_start_frame(strategy=self._start_frame_dropdown.value), start_place


class NormalizingWidget(_LogicWidgetBase):
    """
    A logic widget for normalizing intensities.
    """

    @staticmethod
    def name() -> str:
        return 'Normalizing widget'

    def __init__(self, min_v: float = 0, max_v: float = 1, **kwargs):
        super().__init__(**kwargs)

        self._min_float = widgets.FloatText(value=min_v, description='min: ')
        self._min_float.add_class('numeric-input')
        self._max_float = widgets.FloatText(value=max_v, description='max: ')
        self._max_float.add_class('numeric-input')

        self.children = [self._min_float, self._max_float]

    def get_tooltip(self) -> str:
        return 'Rescales the intensities linearly to be in [min, max].'

    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        return darr.normalized(self._min_float.value, self._max_float.value), start_frame, start_place


class ClippingWidget(_LogicWidgetBase):
    """
    A logic widget for clipping intensity values.
    """

    @staticmethod
    def name() -> str:
        return 'Clipping widget'

    def __init__(self, low: float = 0.5, high: float = 99.5, **kwargs):
        super().__init__(**kwargs)

        self._low_float = widgets.FloatText(value=low, description='low (%): ')
        self._low_float.add_class('numeric-input')
        self._high_float = widgets.FloatText(value=high, description='high (%): ')
        self._high_float.add_class('numeric-input')

        self.children = [self._low_float, self._high_float]

    def get_tooltip(self) -> str:
        return (
            'Replaces the values of quantiles outside the low and high percentiles with the closest values inside.\n'
            'Percentile calculations are done along all data at once, not frame by frame'
        )

    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        return darr.percentile_clipped(self._low_float.value, self._high_float.value), start_frame, start_place


class BackgroundRemovalWidget(_LogicWidgetBase):
    """
    A logic widget for customizing the background removal process.
    """

    @staticmethod
    def name() -> str:
        return 'Background removing widget'

    def __init__(self, frames_before: int = 3, aggregator: str = 'mean', **kwargs):
        super().__init__(**kwargs)

        self._aggregators_by_name = dict(mean=np.mean, median=np.median)

        self._frames_int_text = widgets.IntText(value=frames_before, description='frames before: ')
        self._frames_int_text.add_class('numeric-input')
        self._aggregator_dropdown = widgets.Dropdown(
            description='aggregator: ',
            options=list(self._aggregators_by_name.keys()),
            value=aggregator
        )
        self.children = [self._frames_int_text, self._aggregator_dropdown]

    def get_tooltip(self) -> str:
        return (
            'Removes the background of the image. You can set how many frames before the start frame should be '
            'considered and also which method to use for calculating the background.'
        )

    def __call__(
            self,
            darr: DiffusionArray,
            start_frame: int,
            start_place: tuple
    ) -> Tuple[DiffusionArray, int, tuple]:
        # noinspection PyTypedDict
        return darr.background_removed(
            background_slices=f'{max(0, start_frame - self._frames_int_text.value)}:{start_frame}',
            aggregator=self._aggregators_by_name[self._aggregator_dropdown.value]
        ), start_frame, start_place
