@0xb5a1c7e0d4f36a29;

# The part of openpilot's cereal/log.capnp (MIT) Diorama reads from a drive's
# rlog: the camera calibration calibrationd logs (liveCalibration upstream,
# extrinsicsCalibration in a fork that renamed it). Every other member of
# Event's union is a Void placeholder, as opendbc's rlog.capnp does it, which
# keeps the layout: the union's members share one pointer and the
# discriminant. ExtrinsicsCalibration keeps all its fields, deprecated ones
# included, so each field sits where cereal puts it.

struct ExtrinsicsCalibration {
  warpMatrixDEPRECATED @0 :List(Float32);
  calStatusDEPRECATED @1 :Int8;
  calCycle @2 :Int32;
  calPerc @3 :Int8;
  extrinsicMatrix @4 :List(Float32);
  warpMatrix2DEPRECATED @5 :List(Float32);
  warpMatrixBigDEPRECATED @6 :List(Float32);
  rpyCalib @7 :List(Float32);
  rpyCalibSpread @8 :List(Float32);
  validBlocks @9 :Int32;
  wideFromDeviceEuler @10 :List(Float32);
  calStatus @11 :Status;
  height @12 :List(Float32);

  enum Status {
    uncalibrated @0;
    calibrated @1;
    invalid @2;
    recalibrating @3;
  }
}

struct Event {
  logMonoTime @0 :UInt64;
  # outside the union in cereal; inside, it would shift the discriminant
  valid @67 :Bool = true;

  union {
    initData @1 :Void;
    frame @2 :Void;
    gpsNMEA @3 :Void;
    sensorEventDEPRECATED @4 :Void;
    can @5 :Void;
    reserved6 @6 :Void;
    reserved7 @7 :Void;
    reserved8 @8 :Void;
    reserved9 @9 :Void;
    reserved10 @10 :Void;
    reserved11 @11 :Void;
    reserved12 @12 :Void;
    reserved13 @13 :Void;
    reserved14 @14 :Void;
    reserved15 @15 :Void;
    reserved16 @16 :Void;
    reserved17 @17 :Void;
    reserved18 @18 :Void;
    calibration @19 :ExtrinsicsCalibration;
    reserved20 @20 :Void;
    reserved21 @21 :Void;
    reserved22 @22 :Void;
    reserved23 @23 :Void;
    reserved24 @24 :Void;
    reserved25 @25 :Void;
    reserved26 @26 :Void;
    reserved27 @27 :Void;
    reserved28 @28 :Void;
    reserved29 @29 :Void;
    reserved30 @30 :Void;
    reserved31 @31 :Void;
    reserved32 @32 :Void;
    reserved33 @33 :Void;
    reserved34 @34 :Void;
    reserved35 @35 :Void;
    reserved36 @36 :Void;
    reserved37 @37 :Void;
    reserved38 @38 :Void;
    reserved39 @39 :Void;
    reserved40 @40 :Void;
    reserved41 @41 :Void;
    reserved42 @42 :Void;
    reserved43 @43 :Void;
    reserved44 @44 :Void;
    reserved45 @45 :Void;
    reserved46 @46 :Void;
    reserved47 @47 :Void;
    reserved48 @48 :Void;
    reserved49 @49 :Void;
    reserved50 @50 :Void;
    reserved51 @51 :Void;
    reserved52 @52 :Void;
    reserved53 @53 :Void;
    reserved54 @54 :Void;
    reserved55 @55 :Void;
    reserved56 @56 :Void;
    reserved57 @57 :Void;
    reserved58 @58 :Void;
    reserved59 @59 :Void;
    reserved60 @60 :Void;
    reserved61 @61 :Void;
    reserved62 @62 :Void;
    reserved63 @63 :Void;
    reserved64 @64 :Void;
    reserved65 @65 :Void;
    reserved66 @66 :Void;
  }
}
