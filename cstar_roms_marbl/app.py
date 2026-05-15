import asyncio
import sys
import typing as t

from cstar.applications.core import (
    ApplicationDefinition,
    RunnerRequest,
    RunnerResult,
    register_application,
)
from cstar.applications.roms_marbl.models import APP_NAME, RomsMarblBlueprint
from cstar.applications.roms_marbl.transforms import RomsMarblTimeSplitter
from cstar.base.exceptions import CstarError
from cstar.base.utils import slugify
from cstar.entrypoint.config import (
    get_job_config,
    get_service_config,
)
from cstar.entrypoint.runner import BlueprintRunner, create_parser
from cstar.execution.handler import ExecutionHandler, ExecutionStatus
from cstar.orchestration.models import (
    Application,
)
from cstar.orchestration.serialization import register_representer, strenum_representer
from cstar.orchestration.transforms import (
    DirectiveConfig,
    OverrideTransform,
)
from cstar.roms import ROMSSimulation

if t.TYPE_CHECKING:
    from cstar.entrypoint.config import JobConfig, ServiceConfiguration


_APP_NAME_LONG: t.Literal["ROMS-MARBL simulation runner"] = (
    "ROMS-MARBL simulation runner"
)


class RomsMarblRunner(BlueprintRunner[RomsMarblBlueprint]):
    """Worker class to run c-star simulations."""

    simulation: t.Final[ROMSSimulation]
    """The simulation instance created from the blueprint."""
    _handler: ExecutionHandler | None = None
    """The execution handler for the simulation."""

    def __init__(
        self,
        request: RunnerRequest[RomsMarblBlueprint],
        service_cfg: "ServiceConfiguration",
        job_cfg: "JobConfig",
    ) -> None:
        """Initialize the RomsMarblRunner with the supplied configuration.

        Parameters
        ----------
        request: RunnerRequest[RomsMarblBlueprint]
            A request containing information about the simulation to run

        service_cfg: ServiceConfiguration
            Configuration for modifying behavior of the service process.

        job_cfg: JobConfig
            Configuration for submitting jobs to an HPC, such as account ID,
            walltime, job name, and priority.
        """
        super().__init__(request, service_cfg, job_cfg)

        self.simulation = ROMSSimulation.from_blueprint(self.request.blueprint_uri)
        self.simulation.name = slugify(self.simulation.name)

    @t.override
    def _on_start(self) -> None:
        super()._on_start()

        if not self.simulation:
            msg = "Simulation creation failed. Unable to execute simulation runner"
            raise RuntimeError(msg)

        self.log.trace("Setting up simulation")
        self.simulation.setup()
        self.log.trace("Building simulation")
        self.simulation.build()
        self.log.trace("Executing simulation pre-run")
        self.simulation.pre_run()

        self.log.trace("Starting simulation.")
        self._handler = self.simulation.run(
            account_key=self._job_cfg.account_id,
            walltime=self._job_cfg.walltime,
            job_name=self._job_cfg.job_name,
        )

    @t.override
    async def run(self) -> RunnerResult[RomsMarblBlueprint]:
        """Execute the c-star simulation."""
        if self._handler is None:
            msg = "Simulation did not start up successfully"
            self.add_state(ExecutionStatus.FAILED, msg)
            raise CstarError(msg)

        try:
            await self._handler.updates(seconds=1.0)
            self.add_state(self._handler.status)
        except Exception as ex:
            msg = "An error occurred while running the simulation"
            self.log.exception(msg)
            self.add_state(ExecutionStatus.FAILED, [msg, str(ex)])

        return self.result

    @t.override
    def _on_iteration_complete(self) -> None:
        """Perform post-processing after each iteration of the main event loop."""
        super()._on_iteration_complete()
        if self.state.status == ExecutionStatus.COMPLETED:
            self.simulation.post_run()
        elif ExecutionStatus.is_terminal(self.state.status):
            msg = "Skipping simulation post-run; simulation did not complete."
            self.log.debug(msg)


@register_application
class RomsMarblApplication(ApplicationDefinition[RomsMarblBlueprint, RomsMarblRunner]):
    name: str = APP_NAME
    long_name: str = _APP_NAME_LONG
    runner = RomsMarblRunner
    blueprint = RomsMarblBlueprint
    applicable_transforms = (RomsMarblTimeSplitter, OverrideTransform)
