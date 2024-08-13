# this file contains the kernel written by David Wichmann for advecting with the MR equation using a RK4 scheme which is written for parcels 2 or even 1
# I will use this as inspiration for writing a parcels 3 kernel for advecting with the MR equation. The kernel presented here is for 2d flows. I will 
# generalize this for 3d flows
# source: 


def MRslow2D_RK4_beaching(particle, fieldset, time, dt):
    """
    Kernel for advection with the slow Maxey-Riley equation (inertial equation) with inertia and Coriolis force for 2D flows
    Advection only happens if the particle is not beached. For this, need an fieldset.Land field defined
    """
    ll = fieldset.Land[0,particle.lon,particle.lat,0]
    
    if ll > 0.0:
        particle.lon+=0
        particle.lat+=0
    else:
        depth = particle.depth
        Omega = 7.2921e-5
        deg2rd = math.pi/180.0
    
        lon1 = particle.lon
        lat1 = particle.lat

        u1 = fieldset.U[time, lon1, lat1, depth]
        v1 = fieldset.V[time, lon1, lat1, depth]        
        dudx1=(fieldset.U[time,lon1+fieldset.Dphi,lat1,depth] - fieldset.U[time,lon1-fieldset.Dphi,lat1,depth])/(2*fieldset.Dphi)
        dudy1=(fieldset.U[time,lon1,lat1+fieldset.Dtheta,depth] - fieldset.U[time,lon1,lat1-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)        
        dvdx1=(fieldset.V[time,lon1+fieldset.Dphi,lat1,depth] - fieldset.V[time,lon1-fieldset.Dphi,lat1,depth])/(2*fieldset.Dphi)        
        dvdy1=(fieldset.V[time,lon1,lat1+fieldset.Dtheta,depth] - fieldset.V[time,lon1,lat1-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)                
        dudt1=(fieldset.U[time+fieldset.Dt,lon1,lat1,depth] - fieldset.U[time-fieldset.Dt,lon1,lat1,depth])/(2*fieldset.Dt)                
        dvdt1=(fieldset.V[time+fieldset.Dt,lon1,lat1,depth] - fieldset.V[time-fieldset.Dt,lon1,lat1,depth])/(2*fieldset.Dt)        
        
        kx1=u1+fieldset.tau*(dudt1+u1*dudx1+v1*dudy1-2*Omega*math.sin(deg2rd*lat1)*v1)        
        ky1=v1+fieldset.tau*(dvdt1+u1*dvdx1+v1*dvdy1+2*Omega*math.sin(deg2rd*lat1)*u1)
    
        lon2 = particle.lon+.5*dt*kx1
        lat2 = particle.lat+.5*dt*ky1

        u2 = fieldset.U[time+.5*dt, lon2, lat2, depth]
        v2 = fieldset.V[time+.5*dt, lon2, lat2, depth]
        dudx2=(fieldset.U[time+.5*dt,lon2+fieldset.Dphi,lat2,depth] - fieldset.U[time+.5*dt,lon2-fieldset.Dphi,lat2,depth])/(2*fieldset.Dphi)
        dudy2=(fieldset.U[time+.5*dt,lon2,lat2+fieldset.Dtheta,depth] - fieldset.U[time+.5*dt,lon2,lat2-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dvdx2=(fieldset.V[time+.5*dt,lon2+fieldset.Dphi,lat2,depth] - fieldset.V[time+.5*dt,lon2-fieldset.Dphi,lat2,depth])/(2*fieldset.Dphi)
        dvdy2=(fieldset.V[time+.5*dt,lon2,lat2+fieldset.Dtheta,depth] - fieldset.V[time+.5*dt,lon2,lat2-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dudt2=(fieldset.U[time+.5*dt+fieldset.Dt,lon2,lat2,depth]-fieldset.U[time+.5*dt-fieldset.Dt,lon2,lat2,depth])/(2*fieldset.Dt)
        dvdt2=(fieldset.V[time+.5*dt+fieldset.Dt,lon2,lat2,depth]-fieldset.V[time+.5*dt-fieldset.Dt,lon2,lat2,depth])/(2*fieldset.Dt)

        kx2=u2+fieldset.tau*(dudt2+u2*dudx2+v2*dudy2-2*Omega*math.sin(deg2rd*lat2)*v2)
        ky2=v2+fieldset.tau*(dvdt2+u2*dvdx2+v2*dvdy2+2*Omega*math.sin(deg2rd*lat2)*u2)
        
        lon3 = particle.lon+.5*dt*kx2
        lat3 = particle.lat+.5*dt*ky2

        u3 = fieldset.U[time+.5*dt, lon3, lat3, depth]
        v3 = fieldset.V[time+.5*dt, lon3, lat3, depth]
        dudx3=(fieldset.U[time+.5*dt,lon3+fieldset.Dphi,lat3,depth] - fieldset.U[time+.5*dt,lon3-fieldset.Dphi,lat3,depth])/(2*fieldset.Dphi)
        dudy3=(fieldset.U[time+.5*dt,lon3,lat3+fieldset.Dtheta,depth] - fieldset.U[time+.5*dt,lon3,lat3-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dvdx3=(fieldset.V[time+.5*dt,lon3+fieldset.Dphi,lat3,depth] - fieldset.V[time+.5*dt,lon3-fieldset.Dphi,lat3,depth])/(2*fieldset.Dphi)
        dvdy3=(fieldset.V[time+.5*dt,lon3,lat3+fieldset.Dtheta,depth] - fieldset.V[time+.5*dt,lon3,lat3-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dudt3=(fieldset.U[time+.5*dt+fieldset.Dt,lon3,lat3,depth]-fieldset.U[time+.5*dt-fieldset.Dt,lon3,lat3,depth])/(2*fieldset.Dt)
        dvdt3=(fieldset.V[time+.5*dt+fieldset.Dt,lon3,lat3,depth]-fieldset.V[time+.5*dt-fieldset.Dt,lon3,lat3,depth])/(2*fieldset.Dt)

        kx3=u3+fieldset.tau*(dudt3+u3*dudx3+v3*dudy3-2*Omega*math.sin(deg2rd*lat3)*v3)
        ky3=v3+fieldset.tau*(dvdt3+u3*dvdx3+v3*dvdy3+2*Omega*math.sin(deg2rd*lat3)*u3)
    
        lon4 = particle.lon+dt*kx3
        lat4 = particle.lat+dt*ky3

        u4 = fieldset.U[time+dt, lon4, lat4, depth]
        v4 = fieldset.V[time+dt, lon4, lat4, depth]
        dudx4=(fieldset.U[time+dt,lon4+fieldset.Dphi,lat4,depth]-fieldset.U[time+dt,lon4-fieldset.Dphi,lat4,depth])/(2*fieldset.Dphi)
        dudy4=(fieldset.U[time+dt,lon4,lat4+fieldset.Dtheta,depth]-fieldset.U[time+dt,lon4,lat4-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dvdx4=(fieldset.V[time+dt,lon4+fieldset.Dphi,lat4,depth]-fieldset.V[time+dt,lon4-fieldset.Dphi,lat4,depth])/(2*fieldset.Dphi)
        dvdy4=(fieldset.V[time+dt,lon4,lat4+fieldset.Dtheta,depth]-fieldset.V[time+dt,lon4,lat4-fieldset.Dtheta,depth])/(2*fieldset.Dtheta)
        dudt4=(fieldset.U[time+dt+fieldset.Dt,lon4,lat4,depth]-fieldset.U[time+dt-fieldset.Dt,lon4,lat4,depth])/(2*fieldset.Dt)
        dvdt4=(fieldset.V[time+dt+fieldset.Dt,lon4,lat4,depth]-fieldset.V[time+dt-fieldset.Dt,lon4,lat4,depth])/(2*fieldset.Dt)

        kx4=u4+fieldset.tau*(dudt4+u4*dudx4+v4*dudy4-2*Omega*math.sin(deg2rd*lat4)*v4)
        ky4=v4+fieldset.tau*(dvdt4+u4*dvdx4+v4*dvdy4+2*Omega*math.sin(deg2rd*lat4)*u4)
    
        particle.lon += (kx1 + 2*kx2 + 2*kx3 + kx4) / 6. * dt
        particle.lat += (ky1 + 2*ky2 + 2*ky3 + ky4) / 6. * dt
        