/**
 * Angular NgModule for any-compute — import in AppModule to provide AnyComputeService.
 * Auto-generated — edit FfiRegistry, not this file.
 */
import { NgModule, APP_INITIALIZER } from '@angular/core';
import { AnyComputeService } from './any-compute.service';

export function initFactory(svc: AnyComputeService) {
  return () => svc.init();
}

@NgModule({
  providers: [
    AnyComputeService,
    { provide: APP_INITIALIZER, useFactory: initFactory, deps: [AnyComputeService], multi: true },
  ],
})
export class AnyComputeModule {}
