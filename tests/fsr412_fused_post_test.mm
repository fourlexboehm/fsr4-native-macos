// Experimental SIMD POST fusion must preserve temporal state and output.
#define FSR412_TEST_HELPERS_ONLY
#include "fsr412_optimization_test.mm"

int main(int argc,char **argv) { @autoreleasepool {try {
    require(argc>=3,"usage: fsr412_fused_post_test <root> <builddir> [mtl4]");
    const bool metal4=argc>3 && !std::strcmp(argv[3],"mtl4");
    auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];
    require(dev && queue,"Metal unavailable");
    fsr412::Runtime runtime(dev,argv[1],argv[2]);
    unsigned failures=0;
    auto check=[&](id<MTLTexture> a,id<MTLTexture> b,unsigned bytes,const char *name,bool exact=false) {
        try {compare(a,b,bytes,name,exact);}
        catch(const std::exception &e) {++failures;std::fprintf(stderr,"PARITY FAILURE: %s\n",e.what());}
    };
    for(unsigned mode=0;mode<2;++mode) {
        fsr412::View baseline,fused;
        fsr412::Options options;options.backend=fsr412::Backend::Simd;
        for(unsigned index=0;index<6;++index) {
            auto a=frame(dev,index,mode==1),b=a;
            b.output=texture(dev,MTLPixelFormatRGBA16Float,a.output.width,a.output.height);
            std::vector<_Float16> sentinel(a.output.width*a.output.height*4,(_Float16)-9);
            for(auto t:{a.output,b.output}) [t replaceRegion:MTLRegionMake2D(0,0,t.width,t.height)
                mipmapLevel:0 withBytes:sentinel.data() bytesPerRow:t.width*8];
            options.rcas=index%2;options.sharpness=.498f;
            // Also exercise a different tensor tier without changing the game.
            options.tier=index>=4?2160:1080;
            options.fusedSimdPost=false;encode(runtime,dev,queue,baseline,a,options,metal4);
            options.fusedSimdPost=true;encode(runtime,dev,queue,fused,b,options,metal4);
            std::printf("checking fused mode=%u frame=%u API=%s\n",mode,index,metal4?"Metal4":"classic");
            check(a.output,b.output,8,"fused output");
            check(baseline.history,fused.history,8,"fused history");
            check(baseline.recurrent,fused.recurrent,4,"fused recurrent",true);
            check(baseline.autoExposure,fused.autoExposure,4,"fused exposure",true);
            require(fused.variantKey.find("_simdfused1")!=std::string::npos,"fusion not selected");
            require(fused.neuralRuns==index+1,"fusion skipped inference");
        }
    }
    require(failures==0,std::to_string(failures)+" SIMD fusion parity checks failed; do not enable in games");
    puts("PASS: SIMD fusion, subrects, odd sizes, tier changes, resets, exposure, motion, RCAS and history");
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"FSR SIMD fusion: %s\n",e.what());return 1;}} }
